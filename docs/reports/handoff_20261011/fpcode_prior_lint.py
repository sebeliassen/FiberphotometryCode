from pathlib import Path
import io
import json
import subprocess
import tarfile
import tempfile
root = Path('/home/fsp585/Desktop/FiberphotometryCode')
python = '/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python'
tracked = subprocess.check_output(['git', 'ls-files', '-z', 'studies', 'tests', 'ops'], cwd=root).decode().split('\0')
paths = [p for p in tracked if p.endswith('.py')]
with tempfile.TemporaryDirectory(prefix='fpcode-lint-before-') as directory:
    before = Path(directory)
    archive = subprocess.check_output(['git', 'archive', '--format=tar', 'HEAD', '--', *paths], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        for member in handle:
            if member.isfile():
                target = before / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(handle.extractfile(member).read())
    # Apply current lint rules to all old source, including formerly excluded tests.
    # These are pre-existing diagnostics rather than newly introduced file moves.
    (before / 'pyproject.toml').write_text((root / 'pyproject.toml').read_text())
    result = subprocess.run([python, '-m', 'ruff', 'check', '--output-format', 'json', 'studies', 'tests', 'ops'], cwd=before, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    items = json.loads(result.stdout)
    for item in items:
        item['filename'] = str(Path(item['filename']).relative_to(before))
    (Path('/tmp') / 'fpcode-prior-lint.json').write_text(json.dumps(items))
    print(f'Captured {len(items)} pre-existing lint diagnostics across {len(paths)} tracked Python files.')
