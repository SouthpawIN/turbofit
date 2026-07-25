FROM python:3.12-slim
WORKDIR /app
COPY control_plane.py /app/control_plane.py
COPY hardware_probe.py /app/hardware_probe.py
COPY manifest.json /etc/turbofit/manifest.json
COPY state.json /var/lib/turbofit/state.json
EXPOSE 8091
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8091/health', timeout=2)"
ENTRYPOINT ["python", "/app/control_plane.py"]