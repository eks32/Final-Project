#Run my API

library(plumber)
r <- plumb("Notes/API/myAPI.R")

#run it on the port in the Dockerfile
r$run(port=8000)
