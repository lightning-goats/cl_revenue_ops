# Package an exact Revenue Ops source checkout without changing the pinned
# CLN, CLBOSS, xrebalance, Python environment or lab wrapper in the base image.
ARG BASE_IMAGE=cl-revenue-ops-grand-prix-base:2987608
FROM ${BASE_IMAGE}

ARG REVENUE_SOURCE_REVISION
ARG EXPERIMENT_PATCH_DIGEST
ARG EXPERIMENT_NAME
RUN test -n "${REVENUE_SOURCE_REVISION}" \
    && test -n "${EXPERIMENT_PATCH_DIGEST}" \
    && test -n "${EXPERIMENT_NAME}"

COPY cl-revenue-ops.py /opt/cl_revenue_ops/cl-revenue-ops.py
COPY modules/ /opt/cl_revenue_ops/modules/

LABEL org.opencontainers.image.revision.revenue_ops="${REVENUE_SOURCE_REVISION}"
LABEL org.opencontainers.image.experiment.name="${EXPERIMENT_NAME}"
LABEL org.opencontainers.image.experiment.patch_digest="${EXPERIMENT_PATCH_DIGEST}"
