# start from the rstudio/plumber image
FROM rstudio/plumber

# install the linux libraries needed for plumber
RUN apt-get update -qq && apt-get install -y  libssl-dev  libcurl4-gnutls-dev  libpng-dev pandoc 
    
    
# install plumber, GGally
RUN R -e "install.packages(c('GGally', 'leaflet', 'plumber'))"

# copy myAPI.R from the current directory into the container
COPY myAPI.R myAPI.R

# open port to traffic
EXPOSE 8000

# when the container starts, start the myAPI.R script
ENTRYPOINT ["R", "-e", \
    "pr <- plumber::plumb('myAPI.R'); pr$run(host='0.0.0.0', port=8000)"]
