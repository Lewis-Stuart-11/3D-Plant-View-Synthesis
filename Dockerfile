FROM dromni/nerfstudio:1.1.3 AS base
WORKDIR /pvs
ADD --checksum=sha256:80c826a115b8feeffe61a5023a8577a0c48dd2e7666a5579a865d515481c4ff3 https://cvl.cs.nott.ac.uk/resources/nerf_data/bc1_1033_3.zip /pvs/
USER root
RUN apt-get -y update \
    && apt-get -y autoremove \
    && apt-get clean \
    && apt-get install -y zip
RUN chown -R user .
USER user
RUN unzip bc1_1033_3.zip
RUN rm bc1_1033_3.zip
COPY *.py *.json *.txt /pvs/
ENTRYPOINT bash