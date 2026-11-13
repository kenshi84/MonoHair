"""COLMAP/FLAME input adapter. No reconstruction or GPU dependencies at import time."""

import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
from PIL import Image, ImageFilter
import yaml


REPO = Path(__file__).resolve().parents[1]
PRESERVED_FILES = ('head_flame.obj', 'sparse/cameras.bin', 'sparse/images.bin',
                   'sparse/points3D.bin')


def _unpack(stream, fmt):
    fmt = '<' + fmt
    data = stream.read(struct.calcsize(fmt))
    if len(data) != struct.calcsize(fmt):
        raise ValueError(f'Truncated COLMAP file: {stream.name}')
    return struct.unpack(fmt, data)


def read_colmap(sparse):
    """Read pinhole calibration; COLMAP stores scalar-first quaternions and w2c poses."""
    cameras, images = {}, []
    with (Path(sparse) / 'cameras.bin').open('rb') as f:
        for _ in range(_unpack(f, 'Q')[0]):
            cid, model, width, height = _unpack(f, 'iiQQ')
            if model == 0:  # SIMPLE_PINHOLE
                focal, cx, cy = _unpack(f, 'ddd')
                intrinsics = [focal, focal, cx, cy]
            elif model == 1:  # PINHOLE
                intrinsics = list(_unpack(f, 'dddd'))
            else:
                raise ValueError(f'Unsupported COLMAP camera model {model} (camera {cid}); '
                                 'expected PINHOLE or SIMPLE_PINHOLE, without distortion')
            if cid in cameras or min(width, height, *intrinsics[:2]) <= 0 or not np.isfinite(intrinsics).all():
                raise ValueError(f'Invalid or duplicate camera {cid}')
            cameras[cid] = dict(width=width, height=height, intrinsics=intrinsics)
    with (Path(sparse) / 'images.bin').open('rb') as f:
        for _ in range(_unpack(f, 'Q')[0]):
            iid, w, x, y, z, tx, ty, tz, cid = _unpack(f, 'idddddddi')
            name = bytearray()
            while True:
                byte = f.read(1)
                if not byte:
                    raise ValueError('Truncated COLMAP image name')
                if byte == b'\0':
                    break
                name.extend(byte)
            name = name.decode('utf8')
            count = _unpack(f, 'Q')[0]
            end = f.tell() + count * 24  # x, y, point3D_id
            if end > Path(f.name).stat().st_size:
                raise ValueError('Truncated COLMAP image observations')
            f.seek(end)
            if cid not in cameras or Path(name).name != name or Path(name).suffix.lower() != '.jpg':
                raise ValueError(f'Invalid image/camera mapping: {name}, camera {cid}')
            if not np.isfinite([w, x, y, z, tx, ty, tz]).all() or not np.isclose(w*w+x*x+y*y+z*z, 1, atol=1e-5):
                raise ValueError(f'Invalid camera pose for {name}')
            rotation = np.array([
                [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
            ])
            images.append(dict(id=iid, name=name, camera_id=cid, rotation=rotation,
                               translation=np.array([tx, ty, tz])))
    if not images or len({im['id'] for im in images}) != len(images) or len({Path(im['name']).stem for im in images}) != len(images):
        raise ValueError('Empty model or duplicate image IDs/names')
    return cameras, sorted(images, key=lambda im: (not Path(im['name']).stem.isdigit(),
                           int(Path(im['name']).stem) if Path(im['name']).stem.isdigit() else im['name']))


def read_obj(path):
    """Keep OBJ vertex indices, including vertices on UV seams (no mesh processing)."""
    vertices, faces, uvs, uv_faces = [], [], [], []
    with Path(path).open() as f:
        for line in f:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == 'v':
                vertices.append([float(x) for x in fields[1:4]])
            elif fields[0] == 'vt':
                uvs.append([float(x) for x in fields[1:3]])
            elif fields[0] == 'f':
                if len(fields) != 4:
                    raise ValueError(f'Expected triangular FLAME faces in {path}')
                corners = [field.split('/') for field in fields[1:]]
                faces.append([int(c[0]) - 1 for c in corners])
                if all(len(c) > 1 and c[1] for c in corners):
                    uv_faces.append([int(c[1]) - 1 for c in corners])
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or not faces.size or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f'Invalid mesh: {path}')
    return vertices, faces, np.asarray(uvs), np.asarray(uv_faces, dtype=np.int64)


def write_obj(path, vertices, faces):
    """Write explicit vertex normals so Open3D can sample scalp root directions."""
    normals = np.zeros_like(vertices)
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                            vertices[faces[:, 2]] - vertices[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    with Path(path).open('w') as f:
        for v in vertices:
            f.write('v ' + ' '.join(format(x, '.17g') for x in v) + '\n')
        for n in normals:
            f.write('vn ' + ' '.join(format(x, '.17g') for x in n) + '\n')
        for face in faces + 1:
            f.write('f ' + ' '.join(f'{i}//{i}' for i in face) + '\n')


def scalp_topology(template, mask_path):
    vertices, faces, uvs, uv_faces = template
    # The existing scalp extraction examines a small neighborhood around each UV.
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert('L').filter(ImageFilter.MaxFilter(7)))
    height, width = mask.shape
    xy = np.rint(uvs * [width-1, height-1]).astype(int)
    xy = np.clip(xy, 0, [width-1, height-1])
    covered = mask[height-1-xy[:, 1], xy[:, 0]] > 0
    selected = np.zeros(len(vertices), dtype=bool)
    # A FLAME vertex can have several UVs; combine them without splitting vertices.
    np.logical_or.at(selected, faces.ravel(), covered[uv_faces.ravel()])
    scalp_faces = faces[selected[faces].all(axis=1)]
    vertex_ids, remapped = np.unique(scalp_faces, return_inverse=True)
    if not len(vertex_ids):
        raise ValueError('The template scalp mask selected no faces')
    return vertex_ids, remapped.reshape(-1, 3)


def transform_points(points, matrix):
    return np.asarray(points) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def camera_parameters(camera, image, source_to_internal, image_scale):
    width = max(1, round(camera['width'] * image_scale))
    height = max(1, round(camera['height'] * image_scale))
    fx, fy, cx, cy = np.asarray(camera['intrinsics']) * [
        width/camera['width'], height/camera['height'],
        width/camera['width'], height/camera['height']]
    pose = np.eye(4)
    rotation = image['rotation']
    # OpenCV -> OpenGL camera basis only; the scene's world axes do not rotate.
    pose[:3, :3] = rotation.T @ np.diag([1, -1, -1])
    pose[:3, 3] = transform_points(-rotation.T @ image['translation'], source_to_internal)
    ndc = [2*fx/width, 2*fy/height, 1-2*cx/width, 2*cy/height-1]
    return dict(file=Path(image['name']).stem, pose=pose.tolist(),
                intrin=[0., 0., 0., 0.], intrin_op=[fx, fy, cx, cy],
                ndc_prj=ndc, dist=[0.] * 5, width=width, height=height)


def inspect_case(case_dir, image_scale=0.5):
    case_dir = Path(case_dir)
    if not np.isfinite(image_scale) or not 0 < image_scale <= 1:
        raise ValueError('image-scale must be in (0, 1]')
    for name in PRESERVED_FILES:
        if not (case_dir / name).is_file():
            raise FileNotFoundError(case_dir / name)
    cameras, images = read_colmap(case_dir / 'sparse')
    names = {im['name'] for im in images}
    if names != {p.name for p in (case_dir / 'cropped/images').glob('*.jpg')}:
        raise ValueError('COLMAP image names must exactly match cropped/images/*.jpg')
    sizes = set()
    for im in images:
        cam = cameras[im['camera_id']]
        with Image.open(case_dir / 'cropped/images' / im['name']) as image:
            if image.size != (cam['width'], cam['height']):
                raise ValueError(f'Image/calibration dimension mismatch: {im["name"]}')
        sizes.add((cam['width'], cam['height']))
    if len(sizes) != 1:
        raise ValueError('PMVO requires all case images to have the same dimensions')
    vertices, faces, _, _ = read_obj(case_dir / 'head_flame.obj')
    template = read_obj(REPO / 'assets/data/head_template.obj')
    if vertices.shape != template[0].shape or not np.array_equal(faces, template[1]):
        raise ValueError('head_flame.obj must retain the template FLAME vertex and face ordering')
    scalp_ids, scalp_faces = scalp_topology(template, REPO / 'assets/data/scalp_mask.png')
    base = yaml.safe_load((REPO / 'configs/reconstruct/base.yaml').read_text())
    target = template[0][scalp_ids].mean(axis=0) + base['bust_to_origin']
    matrix = np.eye(4)
    matrix[:3, :3] *= 0.001
    matrix[:3, 3] = target - vertices[scalp_ids].mean(axis=0) * 0.001
    return dict(cameras=cameras, images=images, vertices=vertices, faces=faces,
                scalp_ids=scalp_ids, scalp_faces=scalp_faces, source_to_internal=matrix)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_case(case_dir, output_dir, image_scale, checkpoints):
    data = inspect_case(case_dir, image_scale)
    case_dir, output_dir = Path(case_dir), Path(output_dir)
    work = output_dir / 'work'
    for folder in ('capture_images', 'ours', 'ngp', 'logs'):
        (work / folder).mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name in PRESERVED_FILES:
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case_dir / name, destination)
        checksums[name] = sha256(case_dir / name)
        if sha256(destination) != checksums[name]:
            raise OSError(f'Copy checksum mismatch: {destination}')
    matrix = data['source_to_internal']
    vertices = transform_points(data['vertices'], matrix)
    write_obj(work / 'ours/bust_long_tsfm.obj', vertices, data['faces'])
    write_obj(work / 'ours/scalp_tsfm.obj', vertices[data['scalp_ids']], data['scalp_faces'])
    np.save(work / 'scalp_vertex_ids.npy', data['scalp_ids'])
    np.eye(4, dtype=np.float32).T.tofile(work / 'model_tsfm.dat')
    np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32).tofile(work / 'model_tsfm_semantic.dat')
    cam_list, frames = [], []
    for im in data['images']:
        cam = camera_parameters(data['cameras'][im['camera_id']], im, matrix, image_scale)
        cam_list.append(cam)
        with Image.open(case_dir / 'cropped/images' / im['name']) as image:
            image.convert('RGB').resize((cam['width'], cam['height']), Image.Resampling.LANCZOS).save(
                work / 'capture_images' / (cam['file'] + '.png'))
        fx, fy, cx, cy = cam['intrin_op']
        frames.append(dict(file_path='../capture_images/' + cam['file'] + '.png',
                           transform_matrix=cam['pose'], fl_x=fx, fl_y=fy, cx=cx, cy=cy,
                           w=cam['width'], h=cam['height']))
    (work / 'ours/cam_params.json').write_text(json.dumps(dict(cam_list=cam_list), indent=2))
    (work / 'ngp/transforms.json').write_text(json.dumps(dict(
        scale=1.0, offset=[0.5, 0.5, 0.5], aabb_scale=1, frames=frames), indent=2))
    metadata = dict(input_case=str(case_dir), units='millimeters', axes='+X left, +Y up, +Z front',
                    source_to_internal=matrix.tolist(), internal_to_source=np.linalg.inv(matrix).tolist(),
                    working_units='meters', working_bust_offset=[0, 0, 0], image_scale=image_scale,
                    working_axes='+X left, +Y up, +Z front',
                    ngp_mesh_convention='work/ngp/base.obj stores internal (y,z,x); '
                                        'work/ours/colmap_points.obj restores internal (x,y,z)',
                    image_names=[im['name'] for im in data['images']], input_sha256=checksums,
                    final_strands='connected_strands.hair', working_directory='work')
    (output_dir / 'coordinate_transform.json').write_text(json.dumps(metadata, indent=2))
    config = yaml.safe_load((REPO / 'configs/reconstruct/base.yaml').read_text())
    config.update(synthetic=True, name='synthetic', output_root='results', bust_to_origin=[0, 0, 0])
    config['data'].update(root=str(output_dir), case='work',
                          image_size=[cam_list[0]['height'], cam_list[0]['width']])
    config['camera_path'] = str(REPO / config['camera_path'])
    config['prepare_data'] = {key: key == 'process_imgs' for key in config['prepare_data']}
    config['segment'].update(checkpoints)
    config_path = work / 'case.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def ngp_mesh_to_internal(vertices):
    # Vendored NGP's OBJ exporter undoes scale/offset, but retains the y,z,x permutation.
    return np.asarray(vertices)[:, [2, 0, 1]]


def prepare_ngp_images(work):
    """Use the pipeline's inferred silhouette as alpha; do not reconstruct the backdrop."""
    work = Path(work)
    transforms = json.loads((work / 'ngp/transforms.json').read_text())
    (work / 'ngp/images').mkdir(exist_ok=True)
    for frame in transforms['frames']:
        name = Path(frame['file_path']).name
        mask_path = work / 'mask' / name
        if not mask_path.is_file():
            raise FileNotFoundError(f'Missing silhouette {mask_path}; run the images stage before ngp')
        with Image.open(work / 'ngp' / frame['file_path']) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(f'Silhouette dimensions disagree with calibration: {name}')
            rgba = image.convert('RGBA')
            rgba.putalpha(mask.convert('L'))
            rgba.save(work / 'ngp/images' / name)
        frame['file_path'] = 'images/' + name
    path = work / 'ngp/train_transforms.json'
    path.write_text(json.dumps(transforms, indent=2))
    return path


def export_strands(source, destination, internal_to_source):
    """Keep MonoHair's strand counts/order and convert only its float32 point payload."""
    raw = Path(source).read_bytes()
    if len(raw) < 8:
        raise ValueError('Truncated hair header')
    strands, count = struct.unpack_from('<II', raw)
    header_size = 8 + 2 * strands
    if not strands or not count or len(raw) != header_size + 12 * count:
        raise ValueError('Empty or truncated strand output')
    lengths = np.frombuffer(raw, dtype='<u2', count=strands, offset=8)
    if lengths.sum() != count or np.any(lengths < 2):
        raise ValueError('Invalid strand lengths')
    points = np.frombuffer(raw, dtype='<f4', offset=header_size).reshape(-1, 3)
    converted = transform_points(points.astype(np.float64), internal_to_source).astype('<f4')
    if not np.isfinite(converted).all():
        raise ValueError('Non-finite strand coordinates')
    Path(destination).write_bytes(raw[:header_size] + converted.tobytes())
    return dict(strands=strands, points=count, bounds_mm=[converted.min(0).tolist(), converted.max(0).tolist()])
