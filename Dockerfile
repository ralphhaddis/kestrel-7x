# The match runs this with no network, a read-only root (writable /tmp), 1 CPU, 512 MB, uid 65534,
# and the map mounted at /arena. Standard library only, so nothing to install.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /bot
COPY *.py ./
RUN python3 -m compileall -q /bot
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
