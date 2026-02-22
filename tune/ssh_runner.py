# tune/ssh_runner.py

"""
Thin SSH utility that ships a trial config to a remote machine and returns metrics.

Credentials are read from environment variables (nothing hard-coded):
  TUNE_SSH_HOST   — hostname or IP of the remote machine
  TUNE_SSH_USER   — SSH username
  TUNE_SSH_REPO   — absolute path to the portfolio repo on the remote machine
  TUNE_SSH_KEY    — (optional) path to private key file; falls back to ssh-agent / default keys

Usage:
    runner = SSHRunner.from_env()
    metrics = runner.run_trial({"value_weight": 0.6, "momentum_weight": 0.4, ...})
"""

import json
import os
import sys


class SSHRunner:
    def __init__(
        self,
        host: str,
        user: str,
        repo_path: str,
        key_path: str | None = None,
    ):
        self.host = host
        self.user = user
        self.repo_path = repo_path
        self.key_path = key_path

    @classmethod
    def from_env(cls) -> "SSHRunner":
        """Build an SSHRunner from environment variables."""
        host = os.environ.get("TUNE_SSH_HOST")
        user = os.environ.get("TUNE_SSH_USER")
        repo = os.environ.get("TUNE_SSH_REPO")
        key  = os.environ.get("TUNE_SSH_KEY")

        if not host or not user or not repo:
            raise EnvironmentError(
                "Missing required env vars: TUNE_SSH_HOST, TUNE_SSH_USER, TUNE_SSH_REPO"
            )

        return cls(host=host, user=user, repo_path=repo, key_path=key)

    def run_trial(self, config_dict: dict) -> dict:
        """
        Send config_dict to the remote machine, run tune/worker.py, and return the
        parsed metrics dict.

        Steps:
          1. Open SSH connection via paramiko
          2. Pipe JSON config to stdin of `python tune/worker.py`
          3. Read stdout, parse as JSON, return metrics dict
          4. Raise RuntimeError on non-zero exit or JSON parse failure
        """
        import paramiko  # deferred so the module can be imported without paramiko installed

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: dict = dict(username=self.user, hostname=self.host)
        if self.key_path:
            connect_kwargs["key_filename"] = self.key_path

        try:
            client.connect(**connect_kwargs)

            command = (
                f"cd {self.repo_path} && "
                f"python tune/worker.py"
            )
            stdin, stdout, stderr = client.exec_command(command)

            # Send the config JSON on stdin and close the channel's write side
            payload = json.dumps(config_dict)
            stdin.write(payload)
            stdin.channel.shutdown_write()

            # Collect outputs
            out_bytes = stdout.read()
            err_bytes = stderr.read()
            exit_status = stdout.channel.recv_exit_status()

            # Forward worker stderr to our stderr for visibility
            if err_bytes:
                sys.stderr.buffer.write(err_bytes)
                sys.stderr.flush()

            if exit_status != 0:
                raise RuntimeError(
                    f"Worker exited with status {exit_status}. "
                    f"Stderr: {err_bytes.decode(errors='replace')[:500]}"
                )

            try:
                metrics = json.loads(out_bytes.decode())
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Could not parse worker JSON output: {exc}. "
                    f"Raw output: {out_bytes[:200]!r}"
                ) from exc

            return metrics

        finally:
            client.close()
