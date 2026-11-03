# A thin layer over Strata's published image so it can be rented on Vast.ai.
#
# Strata's image sets HOME=/data/home, a directory that does not exist when the
# image is built. Vast wraps every rented image in an SSH layer whose build step
# appends to ~/.bashrc, and that step fails when HOME points nowhere, so the
# host never comes up. Resetting HOME is the whole change: the bench's launch
# script exports Strata's own environment explicitly, so nothing else in the
# image needs to differ.
ARG BASE
FROM ${BASE}
ENV HOME=/root
RUN mkdir -p /root
LABEL org.opencontainers.image.source="https://github.com/zeeshanhaque21/larri" \
      org.opencontainers.image.description="Strata with HOME reset so Vast.ai can wrap it for SSH"
