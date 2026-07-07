"""Upload CloakBrowser to remote server and install Python 3.11.

FIXED (review finding #10): this used to hardcode a plaintext root SSH
password directly in source, plus AutoAddPolicy (no host-key
verification). Credentials now come from environment variables and the
script refuses to run without them.

IMPORTANT: this file used to hardcode a plaintext root SSH password
directly in source. That password is exposed in this file's git history.
If it is still active on the target host, rotate it -- removing it from
this file does not undo that exposure.

Usage:
    CLOAK_DEPLOY_PASSWORD=... python3 deploy_cloak.py
    # optionally override CLOAK_DEPLOY_HOST / CLOAK_DEPLOY_PORT / CLOAK_DEPLOY_USER
    # set CLOAK_DEPLOY_ALLOW_UNKNOWN_HOST=1 only for trusted internal hosts
"""
import paramiko, time, os, tarfile, io

SSH_HOST = os.environ.get("CLOAK_DEPLOY_HOST", "192.168.124.2")
SSH_PORT = int(os.environ.get("CLOAK_DEPLOY_PORT", "2223"))
SSH_USER = os.environ.get("CLOAK_DEPLOY_USER", "root")
SSH_PASSWORD = os.environ.get("CLOAK_DEPLOY_PASSWORD")
ALLOW_UNKNOWN_HOST = os.environ.get("CLOAK_DEPLOY_ALLOW_UNKNOWN_HOST") == "1"

if not SSH_PASSWORD:
    raise SystemExit(
        "CLOAK_DEPLOY_PASSWORD environment variable is required -- "
        "credentials must not be hardcoded in source. "
        "Run as: CLOAK_DEPLOY_PASSWORD=... python3 deploy_cloak.py"
    )

client = paramiko.SSHClient()
client.load_system_host_keys()
if ALLOW_UNKNOWN_HOST:
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
else:
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD,
               timeout=15, look_for_keys=False, allow_agent=False)

def run(cmd):
    stdin, stdout, stderr = client.exec_command(cmd)
    return stdout.read().decode('utf-8', errors='replace')

# Step 1: Install Python 3.11
print("=== Install Python 3.11 ===")
run("cd /tmp/Python-3.11.9 && make altinstall 2>&1 | tail -3")
print(run("/usr/local/python3.11/bin/python3.11 --version"))
pip_check = run("/usr/local/python3.11/bin/pip3.11 --version 2>&1")
print(pip_check)
if "No module named" in pip_check:
    run("/usr/local/python3.11/bin/python3.11 -m ensurepip 2>&1")
    print(run("/usr/local/python3.11/bin/pip3.11 --version 2>&1"))

# Step 2: Create tar.gz of CloakBrowser (excluding .git, __pycache__)
print("\n=== Creating CloakBrowser tar.gz ===")
local_cloak = os.path.normpath(r'E:\CloakBrowser')
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as tar:
    for root, dirs, files in os.walk(local_cloak):
        # Skip .git and __pycache__
        rel = os.path.relpath(root, local_cloak)
        if '.git' in rel.split(os.sep):
            continue
        if '__pycache__' in rel.split(os.sep):
            continue
        for f in files:
            if f.endswith('.pyc'):
                continue
            full = os.path.join(root, f)
            arcname = os.path.relpath(full, local_cloak).replace('\\', '/')
            tar.add(full, arcname=arcname)
buf.seek(0)
print(f"tar.gz size: {len(buf.getvalue())} bytes")

# Step 3: Upload
print("Uploading to remote...")
sftp = client.open_sftp()
sftp.putfo(buf, '/tmp/cloakbrowser.tar.gz')
sftp.close()
print("Uploaded")

# Step 4: Extract
print("\n=== Extract ===")
run("rm -rf /tmp/cloakbrowser_src")
run("mkdir -p /tmp/cloakbrowser_src")
run("cd /tmp/cloakbrowser_src && tar xzf /tmp/cloakbrowser.tar.gz 2>&1")
print(run("ls /tmp/cloakbrowser_src/pyproject.toml 2>&1"))

# Step 5: Install CloakBrowser + playwright
print("\n=== pip install cloakbrowser ===")
cmd = "/usr/local/python3.11/bin/pip3.11 install -e /tmp/cloakbrowser_src 2>&1"
stdin, stdout, stderr = client.exec_command(cmd, timeout=180)
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
print(out[-800:])
if err:
    print("STDERR:", err[-500:])

# Step 6: Install playwright browsers (chromium)
print("\n=== Install playwright chromium ===")
cmd = "/usr/local/python3.11/bin/python3.11 -m playwright install chromium 2>&1"
stdin, stdout, stderr = client.exec_command(cmd, timeout=300)
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
print(out[-500:])
if err:
    print("STDERR:", err[-500:])

# Step 7: Install system deps
print("\n=== Install system deps for chromium ===")
deps_cmd = "/usr/local/python3.11/bin/python3.11 -m playwright install-deps chromium 2>&1"
stdin, stdout, stderr = client.exec_command(deps_cmd, timeout=120)
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
print(out)
if err:
    print("STDERR:", err[-500:])

client.close()
print("\nDone")
