FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04
COPY runtime/ /opt/turbofit/runtime/
ENV LD_LIBRARY_PATH=/opt/turbofit/runtime:${LD_LIBRARY_PATH}
COPY entrypoint.sh /usr/local/bin/turbofit-bonsai
RUN chmod 755 /usr/local/bin/turbofit-bonsai && useradd --create-home turbofit
USER turbofit
EXPOSE 11610
HEALTHCHECK --interval=10s --timeout=5s --retries=12 CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/11610'
ENTRYPOINT ["/usr/local/bin/turbofit-bonsai"]