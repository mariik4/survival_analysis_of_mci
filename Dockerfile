FROM python:3.13-slim

# Install Python
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    && apt-get clean

# # Tell R where to install and find packages
# ENV R_LIBS_USER=/usr/local/lib/R/library
# RUN mkdir -p /usr/local/lib/R/library

# Install R packages
COPY r_packages_install.R .
RUN Rscript r_packages_install.R

# Install Python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN pip3 install --upgrade pip setuptools wheel

EXPOSE 8888

COPY . /project
WORKDIR /project

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", "--notebook-dir=/project"]
