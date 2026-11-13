#!/usr/bin/env python3
"""Run a calibrated synthetic case without fitting its cameras or supplied FLAME head."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

import numpy as np
import yaml

from Utils.synthetic_utils import (
    REPO, PRESERVED_FILES, export_strands, inspect_case, ngp_mesh_to_internal,
    prepare_case, prepare_ngp_images, read_obj, sha256, write_obj,
)


STAGES = ('prepare', 'images', 'ngp', 'depth', 'pmvo', 'inner', 'grow', 'export')


def import_ngp():
    # Prefer the repository build, also allowing an installed pyngp/PYTHONPATH build.
    sys.path.insert(0, str(REPO / 'submodules/instant-ngp/build'))
    import pyngp
    return pyngp


def preflight(stages, checkpoints):
    required = []
    if 'images' in stages:
        required.extend(checkpoints.values())
    if 'inner' in stages:
        required.extend(str(REPO / 'submodules/DeepMVSHair/trained/wky10-30-undirection' / name)
                        for name in ('occ.pth', 'ori.pth'))
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError('Missing pretrained checkpoints:\n  ' + '\n  '.join(missing) +
                                '\nSupply --modnet-checkpoint / --cdgnet-checkpoint as needed. '
                                'Input checks and --stage prepare do not require checkpoints.')
    if any(stage in stages for stage in ('images', 'ngp', 'depth', 'pmvo', 'inner', 'grow')):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('The existing reconstruction stages require CUDA')
    if 'ngp' in stages:
        import_ngp()
    if 'depth' in stages or 'inner' in stages:
        import moderngl
        context = moderngl.create_standalone_context(backend='egl')
        context.release()


def train_ngp(work, steps, resolution, threshold):
    from tqdm import tqdm
    ngp = import_ngp()
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.root_dir = str(REPO / 'submodules/instant-ngp')
    transforms = prepare_ngp_images(work)
    testbed.load_training_data(str(transforms))
    testbed.reload_network_from_file(str(REPO / 'submodules/instant-ngp/configs/nerf/base.json'))
    training = testbed.nerf.training
    training.optimize_extrinsics = False
    training.optimize_focal_length = False
    training.optimize_distortion = False
    frames = json.loads(transforms.read_text())['frames']
    count = len(frames)
    poses_before = [training.get_camera_extrinsics(i).copy() for i in range(count)]
    intrinsics_before = [(meta.focal_length.copy(), meta.principal_point.copy())
                         for meta in training.dataset.metadata]
    for i, pose in enumerate(poses_before):
        if not np.allclose(pose, np.asarray(frames[i]['transform_matrix'])[:3], atol=1e-6, rtol=0):
            raise RuntimeError(f'Instant-NGP camera import disagrees with input camera {i}')
        focal, principal = intrinsics_before[i]
        frame = frames[i]
        if (not np.allclose(focal, [frame['fl_x'], frame['fl_y']], atol=0.001, rtol=0)
                or not np.allclose(principal, [frame['cx']/frame['w'], frame['cy']/frame['h']], atol=1e-6, rtol=0)):
            raise RuntimeError(f'Instant-NGP intrinsics disagree with input camera {i}')
    testbed.shall_train = True
    with tqdm(total=steps, desc='Instant-NGP', unit='steps') as progress:
        while testbed.training_step < steps:
            if not testbed.frame():
                raise RuntimeError('Instant-NGP stopped before finishing training')
            progress.update(min(testbed.training_step, steps) - progress.n)
    for i, before in enumerate(poses_before):
        meta = training.dataset.metadata[i]
        if (not np.array_equal(before, training.get_camera_extrinsics(i))
                or not np.array_equal(intrinsics_before[i][0], meta.focal_length)
                or not np.array_equal(intrinsics_before[i][1], meta.principal_point)):
            raise RuntimeError(f'Instant-NGP changed camera {i}')
    testbed.save_snapshot(str(work / 'ngp/base.ingp'), False)
    mesh = work / 'ngp/base.obj'
    # Restrict extraction to the existing PMVO volume, in NGP's y,z,x unit cube.
    # Otherwise unobserved/background surfaces are sampled and clamped onto its edges.
    low, high = np.array([-.32, -.32, -.24]), np.array([.32, .32, .24])
    aabb = ngp.BoundingBox((low + .5)[[1, 2, 0]], (high + .5)[[1, 2, 0]])
    testbed.compute_and_save_marching_cubes_mesh(str(mesh), [resolution] * 3, aabb=aabb, thresh=threshold)
    vertices, faces, _, _ = read_obj(mesh)
    write_obj(work / 'ours/colmap_points.obj', ngp_mesh_to_internal(vertices), faces)
    (work / 'ngp/training_report.json').write_text(json.dumps(dict(
        steps=int(testbed.training_step), mesh_resolution=resolution, cameras_unchanged=True,
        vertices=len(vertices), faces=len(faces)), indent=2))


def local_stage(stage, output, config, args):
    work = output / 'work'
    if stage == 'ngp':
        train_ngp(work, args.ngp_steps, args.mesh_resolution, config['ngp']['marching_cubes_density_thresh'])
    elif stage == 'depth':
        from Utils.Render_utils import render_bust_hair_depth
        render_bust_hair_depth(str(work / 'ours/colmap_points.obj'),
                              str(work / 'ours/cam_params.json'), str(work / 'render_depth'),
                              image_size=config['data']['image_size'], capture_imgs=True,
                              bust_path=str(work / 'ours/bust_long_tsfm.obj'),
                              bust_to_origin=(0, 0, 0))
    elif stage == 'export':
        metadata = json.loads((output / 'coordinate_transform.json').read_text())
        for name in PRESERVED_FILES:
            if sha256(output / name) != metadata['input_sha256'][name]:
                raise ValueError(f'Preserved input was modified: {name}')
            if sha256(Path(metadata['input_case']) / name) != metadata['input_sha256'][name]:
                raise ValueError(f'Source input changed since preparation: {name}')
        source = work / config['output_root'] / config['name'] / 'full/connected_strands.hair'
        report = export_strands(source, output / 'connected_strands.hair', metadata['internal_to_source'])
        (work / 'export_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


def run_logged(command, logfile, work):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               XDG_CACHE_HOME=str(work / 'cache'), TORCH_HOME=str(work / 'cache/torch'),
               CUDA_CACHE_PATH=str(work / 'cache/cuda'))
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print(f'Running {logfile.stem}; log: {logfile}', flush=True)
    with logfile.open('w') as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = '\n'.join(line[-400:] for line in logfile.read_text(errors='replace').splitlines()[-25:])
        raise RuntimeError(f'{logfile.stem} failed (exit {result.returncode}):\n{tail}\nFull log: {logfile}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('--input-root', type=Path, default=REPO / 'input')
    parser.add_argument('--output-root', type=Path, default=REPO / 'output')
    parser.add_argument('--image-scale', type=float, default=0.5)
    parser.add_argument('--modnet-checkpoint', type=Path)
    parser.add_argument('--cdgnet-checkpoint', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Validate inputs without creating files or using CUDA')
    parser.add_argument('--stage', choices=('all',) + STAGES, default='all',
                        help='Run all stages, or one stage for inspection/resuming (prepare first)')
    parser.add_argument('--ngp-steps', type=int, default=35000)
    parser.add_argument('--mesh-resolution', type=int, default=512)
    args = parser.parse_args()
    try:
        case = args.case_dir.resolve()
        relative = case.relative_to(args.input_root.resolve())
        if not relative.parts:
            raise ValueError('case_dir must be a case below input-root')
        output = (args.output_root.resolve() / relative).resolve()
        if output.is_relative_to(args.input_root.resolve()):
            raise ValueError('output-root must be outside the input tree')
        if args.ngp_steps <= 0 or args.mesh_resolution < 2:
            raise ValueError('ngp-steps must be positive and mesh-resolution must be at least 2')
        data = inspect_case(case, args.image_scale)
        if args.check_only:
            print(json.dumps(dict(case=str(args.case_dir), views=len(data['images']),
                                  vertices=len(data['vertices']), faces=len(data['faces']),
                                  output=str(output))))
            return
        stages = STAGES if args.stage == 'all' else (args.stage,)
        os.environ.update(XDG_CACHE_HOME=str(output / 'work/cache'),
                          TORCH_HOME=str(output / 'work/cache/torch'),
                          CUDA_CACHE_PATH=str(output / 'work/cache/cuda'))
        if any(stage in stages for stage in ('pmvo', 'inner')) and len(data['images']) < 20:
            raise ValueError('The existing PMVO optimizer requires at least 20 calibrated images')
        config_path = output / 'work/case.yaml'
        if 'prepare' in stages:
            config = yaml.safe_load((REPO / 'configs/reconstruct/base.yaml').read_text())
        else:
            if not config_path.is_file():
                raise FileNotFoundError(f'Run --stage prepare first: {config_path}')
            config = yaml.safe_load(config_path.read_text())
        checkpoints = {key: str((REPO / config['segment'][key]).resolve())
                       for key in ('MODNET_ckpt', 'CDGNET_ckpt')}
        for key, value in (('MODNET_ckpt', args.modnet_checkpoint), ('CDGNET_ckpt', args.cdgnet_checkpoint)):
            if value is not None:
                checkpoints[key] = str(value.resolve())
        preflight(stages, checkpoints)
        if 'prepare' in stages:
            config_path = prepare_case(case, output, args.image_scale, checkpoints)
            config = yaml.safe_load(config_path.read_text())
        elif checkpoints != {key: config['segment'][key] for key in checkpoints}:
            config['segment'].update(checkpoints)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        if args.stage in ('ngp', 'depth', 'export'):
            # Individual workers are also usable directly for stage inspection.
            os.chdir(REPO)
            local_stage(args.stage, output, config, args)
            return
        scripts = dict(images='prepare_data.py', pmvo='PMVO.py', inner='infer_inner.py', grow='HairGrow.py')
        for stage in stages:
            if stage == 'prepare':
                continue
            if stage in scripts:
                command = [sys.executable, '-B', str(REPO / scripts[stage]), f'--yaml={config_path.with_suffix("")}']
            else:
                command = [sys.executable, '-B', str(REPO / 'run_synthetic.py'), str(case),
                           '--input-root', str(args.input_root.resolve()), '--output-root', str(args.output_root.resolve()),
                           '--stage', stage, '--ngp-steps', str(args.ngp_steps),
                           '--mesh-resolution', str(args.mesh_resolution)]
            run_logged(command, output / 'work/logs' / f'{stage}.log', output / 'work')
        print(f'Finished {args.stage}: {output}', flush=True)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
