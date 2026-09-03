ARG BASE_IMAGE=cl-revenue-ops-grand-prix-base:2987608
FROM ${BASE_IMAGE}

ARG EXPERIMENT_PATCH_DIGEST
ARG EXPERIMENT_NAME=yield-aware
RUN test -n "${EXPERIMENT_PATCH_DIGEST}"

# Keep the equal, pinned CLN/CLBOSS/xrebalance runtime from BASE_IMAGE and
# replace only the Revenue Ops sources under test.
COPY cl-revenue-ops.py /opt/cl_revenue_ops/cl-revenue-ops.py
COPY modules/config.py /opt/cl_revenue_ops/modules/config.py
COPY modules/fee_controller.py /opt/cl_revenue_ops/modules/fee_controller.py

LABEL org.opencontainers.image.experiment.name="${EXPERIMENT_NAME}"
LABEL org.opencontainers.image.experiment.patch_digest="${EXPERIMENT_PATCH_DIGEST}"
