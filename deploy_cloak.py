"""Upload CloakBrowser to remote server and install Python 3.11."""
import paramiko, time, os, tarfile, io

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.124.2', port=2223, username='root', password='cacjszx.132', timeout=15, look_for_keys=False, allow_agent=False)

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
