import json
import argparse
from pathlib import Path
import runpy
import struct
import sys
import types

import numpy as np
from PIL import Image
import pytest
import yaml

from Utils.synthetic_utils import (
    REPO, PRESERVED_FILES, camera_parameters, export_strands, inspect_case,
    ngp_mesh_to_internal, prepare_case, prepare_ngp_images, read_colmap, read_obj, sha256, transform_points,
    write_obj,
)


@pytest.fixture
def case(tmp_path):
    path = tmp_path / 'input/example'
    (path / 'sparse').mkdir(parents=True)
    (path / 'cropped/images').mkdir(parents=True)
    vertices, faces, _, _ = read_obj(REPO / 'assets/data/head_template.obj')
    write_obj(path / 'head_flame.obj', vertices * 1000 + [13, -1500, 24], faces)
    # IDs deliberately differ from filenames and from each other. Principal points
    # are off-center (one outside the crop), and fx differs from fy.
    with (path / 'sparse/cameras.bin').open('wb') as f:
        f.write(struct.pack('<Q', 2))
        f.write(struct.pack('<iiQQdddd', 17, 1, 80, 120, 400, 420, 30, 145))
        f.write(struct.pack('<iiQQddd', 5, 0, 80, 120, 380, 39, 55))
    with (path / 'sparse/images.bin').open('wb') as f:
        f.write(struct.pack('<Q', 2))
        for iid, name, cid in ((32, '9.jpg', 5), (4, '2.jpg', 17)):
            f.write(struct.pack('<idddddddi', iid, 1, 0, 0, 0, 2, -3, 1700, cid))
            f.write(name.encode() + b'\0' + struct.pack('<Q', 0))
            Image.new('RGB', (80, 120), (120, 100, 90)).save(path / 'cropped/images' / name)
    (path / 'sparse/points3D.bin').write_bytes(struct.pack('<Q', 0))
    return path


def test_colmap_mapping_and_validation(case):
    cameras, images = read_colmap(case / 'sparse')
    assert [image['name'] for image in images] == ['2.jpg', '9.jpg']
    assert images[0]['camera_id'] == 17
    assert cameras[5]['intrinsics'] == [380, 380, 39, 55]
    Image.new('RGB', (40, 60)).save(case / 'cropped/images/2.jpg')
    with pytest.raises(ValueError, match='dimension mismatch'):
        inspect_case(case)


def test_reject_unsupported_and_truncated_cameras(case):
    path = case / 'sparse/cameras.bin'
    raw = bytearray(path.read_bytes())
    struct.pack_into('<i', raw, 12, 4)  # OPENCV distortion must not be silently dropped.
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='Unsupported COLMAP'):
        read_colmap(case / 'sparse')
    path.write_bytes(b'\0')
    with pytest.raises(ValueError, match='Truncated'):
        read_colmap(case / 'sparse')


def test_prepare_preserves_head_calibration_and_scalp(case, tmp_path):
    before = {name: sha256(case / name) for name in PRESERVED_FILES}
    paths_before = set(tmp_path.rglob('*'))
    output = tmp_path / 'output/example'
    config_path = prepare_case(case, output, 0.5, dict(MODNET_ckpt='/models/modnet', CDGNET_ckpt='/models/cdgnet'))
    assert all(path.is_relative_to(tmp_path / 'output')
               for path in set(tmp_path.rglob('*')) - paths_before)
    config = yaml.safe_load(config_path.read_text())
    assert config['data']['image_size'] == [60, 40]
    assert config['bust_to_origin'] == [0, 0, 0]
    assert config['prepare_data']['fit_bust'] is False
    assert config['synthetic'] is True
    assert Path(config['data']['root']) / config['data']['case'] == output / 'work'
    for name in PRESERVED_FILES:
        assert before[name] == sha256(case / name) == sha256(output / name)
    meta = json.loads((output / 'coordinate_transform.json').read_text())
    original, faces, _, _ = read_obj(case / 'head_flame.obj')
    working, working_faces, _, _ = read_obj(output / 'work/ours/bust_long_tsfm.obj')
    assert np.array_equal(faces, working_faces)
    np.testing.assert_allclose(transform_points(working, meta['internal_to_source']), original, atol=1e-9)
    scalp, scalp_faces, _, _ = read_obj(output / 'work/ours/scalp_tsfm.obj')
    ids = np.load(output / 'work/scalp_vertex_ids.npy')
    np.testing.assert_allclose(scalp, working[ids], atol=1e-15)
    template, _, _, _ = read_obj(REPO / 'assets/data/head_template.obj')
    np.testing.assert_allclose(scalp.mean(0), template[ids].mean(0) + [0.006, -1.644, 0.010], atol=1e-15)
    assert set(map(tuple, ids[scalp_faces])).issubset(set(map(tuple, faces)))
    transforms = json.loads((output / 'work/ngp/transforms.json').read_text())
    assert len(transforms['frames']) == 2
    assert transforms['frames'][0]['fl_x'] == 200
    assert transforms['frames'][0]['cy'] == 72.5
    for frame in transforms['frames']:
        with Image.open(output / 'work/ngp' / frame['file_path']) as image:
            assert image.size == (frame['w'], frame['h'])


def assert_projection(data, scale):
    torch = pytest.importorskip('torch')
    from Utils.Camera_utils import Camera
    points = data['vertices'][::83]
    matrix = data['source_to_internal']
    working = torch.tensor(transform_points(points, matrix), dtype=torch.float32)
    for im in data['images']:
        original = data['cameras'][im['camera_id']]
        derived = camera_parameters(original, im, matrix, scale)
        cam = Camera(derived['ndc_prj'], np.linalg.inv(derived['pose']), derived['file'])
        uv, _ = cam.projection(working)
        pixels = cam.uv2pixel(uv, [derived['height'], derived['width']], 'cpu').numpy()[:, ::-1]
        camera_points = points @ im['rotation'].T + im['translation']
        fx, fy, cx, cy = original['intrinsics']
        expected = camera_points[:, :2] / camera_points[:, 2:] * [fx, fy] + [cx, cy]
        expected *= [derived['width']/original['width'], derived['height']/original['height']]
        np.testing.assert_allclose(pixels, expected, atol=0.001, rtol=0)


@pytest.mark.parametrize('scale', [0.5, 1.0, 0.637])
def test_projection_including_rounded_resize(case, scale):
    assert_projection(inspect_case(case), scale)


def test_camera_filter_uses_actual_image_membership(tmp_path):
    from Utils.Camera_utils import parsing_camera
    (tmp_path / '2.jpg').touch()
    cameras = [dict(file=name, ndc_prj=[1, 1, 0, 0], pose=np.eye(4).tolist()) for name in ('1', '2')]
    assert list(parsing_camera(cameras, str(tmp_path))) == ['2']


def test_ngp_axes_and_strand_export(tmp_path):
    points = np.array([[0.01, -0.2, 0.03], [0.02, -0.19, 0.04], [0.025, -0.18, 0.05]])
    np.testing.assert_array_equal(ngp_mesh_to_internal(points[:, [1, 2, 0]]), points)
    original = tmp_path / 'internal.hair'
    exported = tmp_path / 'millimeters.hair'
    header = struct.pack('<IIH', 1, 3, 3)
    payload = header + points.astype('<f4').tobytes()
    original.write_bytes(payload)
    inverse = np.eye(4)
    inverse[:3, :3] *= 1000
    inverse[:3, 3] = [-4, 150, 12]
    report = export_strands(original, exported, inverse)
    assert report['strands'] == 1 and report['points'] == 3
    assert original.read_bytes() == payload and exported.read_bytes()[:10] == header
    actual = np.frombuffer(exported.read_bytes()[10:], dtype='<f4').reshape(-1, 3)
    np.testing.assert_allclose(actual, transform_points(points, inverse), atol=0.0001)
    original.write_bytes(struct.pack('<II', 0, 0))
    with pytest.raises(ValueError, match='Empty'):
        export_strands(original, exported, inverse)


def test_ngp_uses_inferred_silhouette_without_changing_cameras(case, tmp_path):
    output = tmp_path / 'output/example'
    prepare_case(case, output, .5, dict(MODNET_ckpt='/models/modnet', CDGNET_ckpt='/models/cdgnet'))
    work = output / 'work'
    original = json.loads((work / 'ngp/transforms.json').read_text())
    with pytest.raises(FileNotFoundError, match='images stage before ngp'):
        prepare_ngp_images(work)
    (work / 'mask').mkdir()
    mask = np.zeros((60, 40), dtype=np.uint8)
    mask[:, 10:30] = 255
    for frame in original['frames']:
        Image.fromarray(mask).save(work / 'mask' / Path(frame['file_path']).name)
    path = prepare_ngp_images(work)
    prepared = json.loads(path.read_text())
    for before, after in zip(original['frames'], prepared['frames']):
        with Image.open(path.parent / after['file_path']) as image:
            np.testing.assert_array_equal(np.asarray(image)[..., 3], mask)
        after['file_path'] = before['file_path']
        assert before == after


CASES = sorted((REPO / 'input/synth/ct2hair').glob('*/head_flame.obj'))


@pytest.mark.parametrize('head', CASES, ids=[p.parent.name for p in CASES])
def test_available_synthetic_cases(head):
    data = inspect_case(head.parent)
    assert len(data['images']) == 80
    assert len(data['vertices']) == 5023
    assert len(data['faces']) == 9976
    assert_projection(data, 0.5)
    matrix = data['source_to_internal']
    np.testing.assert_allclose(transform_points(transform_points(data['vertices'], matrix),
                                               np.linalg.inv(matrix)), data['vertices'], atol=1e-10)


def test_default_capture_configuration_is_unchanged():
    import inspect
    import options
    from Utils.Render_utils import render_bust_hair_depth
    config = yaml.safe_load((REPO / 'configs/reconstruct/base.yaml').read_text())
    assert config['data']['root'] == 'data'
    assert config['bust_to_origin'] == [0.006, -1.644, 0.010]
    assert config['prepare_data']['fit_bust'] is True
    assert config['output_root'] == 'output'
    assert inspect.signature(render_bust_hair_depth).parameters['bust_to_origin'].default == (0.006, -1.644, 0.010)
    for path in (REPO / 'configs/reconstruct').glob('*.yaml'):
        if path.stem == 'base':
            continue
        example = options.load_options(str(path))
        assert example.data.root == 'data'
        assert example.data.case == path.stem
        assert example.bust_to_origin == config['bust_to_origin']
        assert example.output_root == 'output'
        assert example.camera_path == config['camera_path']


def test_inner_inference_forwards_working_case_and_overrides(case, tmp_path, monkeypatch):
    output = tmp_path / 'output/example'
    config_path = prepare_case(case, output, 0.5, dict(MODNET_ckpt='/models/modnet', CDGNET_ckpt='/models/cdgnet'))
    config = yaml.safe_load(config_path.read_text())
    config['infer_inner']['render_data'] = False
    config_path.write_text(yaml.safe_dump(config))
    calls = {}
    fake_mvs = types.ModuleType('mvs_eval')
    fake_mvs.config_ds_real_parser = argparse.ArgumentParser
    fake_mvs.deep_mvs_eval = lambda args: calls.update(mvs=args)
    monkeypatch.setitem(sys.modules, 'mvs_eval', fake_mvs)
    monkeypatch.setattr('subprocess.run', lambda cmd, **kw: calls.update(command=cmd, check=kw['check']))
    monkeypatch.setattr(sys, 'argv', ['infer_inner.py', f'--yaml={config_path.with_suffix("")}', '--PMVO.optimize=true'])
    runpy.run_path(str(REPO / 'infer_inner.py'), run_name='__main__')
    assert calls['mvs'].root_folder == str(output)
    assert calls['mvs'].case == 'work'
    assert calls['mvs'].calib_folder == str(Path(config['camera_path']).parent)
    assert f'--yaml={config_path.with_suffix("")}' in calls['command']
    assert calls['command'].count('--PMVO.optimize=') == 1
    assert '--PMVO.optimize=true' not in calls['command']
    assert calls['check'] is True


def test_preflight_missing_weights_does_not_start_cuda(tmp_path):
    from run_synthetic import preflight
    with pytest.raises(FileNotFoundError, match='Missing pretrained checkpoints'):
        preflight(('images', 'ngp'), {'CDGNET_ckpt': str(tmp_path / 'missing.pth')})
