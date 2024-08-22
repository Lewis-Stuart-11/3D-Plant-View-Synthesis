FROM dromni/nerfstudio:1.1.3 AS base
WORKDIR /pvs
ADD --checksum=sha256:459c79553ab49d72dc32dd4533db69362e9cfa9a4e0af9651330e863b14fbcf0 https://cvl.cs.nott.ac.uk/resources/nerf_data/bc1_1033_3.zip /pvs/
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