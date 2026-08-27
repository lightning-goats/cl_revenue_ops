ARG CLN_IMAGE=elementsproject/lightningd:v26.06.6@sha256:094be3630f865c795649d6063a8796afa0f78e82a0c311bb34f2b0bd570c819a

FROM debian:bookworm AS clboss-builder
ARG CLBOSS_COMMIT=8cb4e9215eba58b049375f234f5f073d0c7fc622
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf autoconf-archive automake ca-certificates g++ git libcurl4-openssl-dev \
        libev-dev libsqlite3-dev libtool libunwind-dev make pkgconf \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/ksedgwic/clboss.git /src/clboss \
    && git -C /src/clboss checkout --detach "${CLBOSS_COMMIT}" \
    && test "$(git -C /src/clboss rev-parse HEAD)" = "${CLBOSS_COMMIT}"
RUN cd /src/clboss \
    && autoreconf -i \
    && ./configure CXXFLAGS="-O2" \
    && make -j"$(nproc)"

FROM rust:1.89-bookworm AS xrebalance-builder
ARG XREBALANCE_COMMIT=fb70bf13cd9f3f79b14100bfdb8f2966884a4142
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/ksedgwic/xrebalance.git /src/xrebalance \
    && git -C /src/xrebalance checkout --detach "${XREBALANCE_COMMIT}" \
    && test "$(git -C /src/xrebalance rev-parse HEAD)" = "${XREBALANCE_COMMIT}"
RUN cd /src/xrebalance && cargo build --release --locked

FROM python:3.11-bookworm AS revenue-builder
ARG REVENUE_COMMIT
RUN test -n "${REVENUE_COMMIT}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/lightning-goats/cl_revenue_ops.git /src/revenue \
    && git -C /src/revenue checkout --detach "${REVENUE_COMMIT}" \
    && test "$(git -C /src/revenue rev-parse HEAD)" = "${REVENUE_COMMIT}"
RUN python -m pip wheel --disable-pip-version-check --wheel-dir=/wheels -r /src/revenue/requirements.txt

FROM ${CLN_IMAGE}
ARG REVENUE_COMMIT
ARG CLBOSS_COMMIT=8cb4e9215eba58b049375f234f5f073d0c7fc622
ARG XREBALANCE_COMMIT=fb70bf13cd9f3f79b14100bfdb8f2966884a4142
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates dnsutils libcurl4 libev4 libsqlite3-0 libunwind8 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=clboss-builder /src/clboss/clboss /usr/local/libexec/clboss
COPY --from=xrebalance-builder /src/xrebalance/target/release/xrebalance /usr/local/libexec/xrebalance
COPY --from=revenue-builder /src/revenue /opt/cl_revenue_ops
COPY --from=revenue-builder /wheels /opt/cl_revenue_ops/wheels
RUN python3 -m venv /opt/cl_revenue_ops/.venv \
    && /opt/cl_revenue_ops/.venv/bin/pip install \
        --disable-pip-version-check --no-index --find-links=/opt/cl_revenue_ops/wheels \
        -r /opt/cl_revenue_ops/requirements.txt \
    && cp /opt/cl_revenue_ops/tools/cl-revenue-ops-polar-wrapper \
        /opt/cl_revenue_ops/cl-revenue-ops-polar-wrapper \
    && chmod 755 \
        /opt/cl_revenue_ops/cl-revenue-ops-polar-wrapper \
        /usr/local/libexec/clboss \
        /usr/local/libexec/xrebalance \
    && rm -rf /opt/cl_revenue_ops/wheels
LABEL org.opencontainers.image.title="Polar CLBOSS competition contender"
LABEL org.opencontainers.image.description="Equal CLN runtime containing pinned cl_revenue_ops, CLBOSS, and xrebalance contenders"
LABEL org.opencontainers.image.revision.revenue_ops="${REVENUE_COMMIT}"
LABEL org.opencontainers.image.revision.clboss="${CLBOSS_COMMIT}"
LABEL org.opencontainers.image.revision.xrebalance="${XREBALANCE_COMMIT}"
