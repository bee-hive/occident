# Processes metadata submissions into web portal md files

# Reads metadata submission tables as TSV files, parses content, and  produces
# .md files into ../_data. Also produces unique .md files for  ../_cell-types 
# and ../_conditions.

library(dplyr)
library(yaml)

## GET TXT FILES
meta.files <- list.files(pattern = ".tsv")

## PROCESS TXT FILES
for(meta.f in meta.files){
  if (meta.f == "TEMPLATE.tsv")
    next
  
  print(meta.f)
  
  ## READ TXT
  meta <- read.table(meta.f, sep = "\t", header = T, stringsAsFactors = F,
                     colClasses = "character")
  template <- read.table("TEMPLATE.tsv", sep = "\t", header = T, 
                         stringsAsFactors = F)
  
  # reformat online submissions
  #TODO
  
  # DATA VALIDATION
  reject.flag <- FALSE
  
  # columns match template
  col.check <- names(meta) == names(template)
  bad.cols <- sum(!col.check, na.rm = FALSE)
  if (bad.cols > 0){
    reject.flag <- TRUE
    print(paste("The submission",meta.f,"has been rejected for the following reasons:"))
    print(paste(bad.cols,"columns do not match those in the template:",
                paste(names(meta)[!col.check],collapse = ", ")))
  }
  # one or more rows
  if(nrow(meta) < 1){
    reject.flag <- TRUE
    print(paste("The submission",meta.f,"has been rejected for the following reasons:"))
    print("No rows detected. The submitted table is empty.")
  }

  
  ## PRODUCE FILES
  if (!reject.flag) {
    for(t in 1:nrow(meta)){
      #TRIALS MD
      md <- meta[t,] %>%
        dplyr::mutate(date = as.character(as.Date(date, "%m/%d/%y"))) %>%
        dplyr::mutate(channels = ifelse(channels != "", strsplit(channels, ","),"")) %>%
        dplyr::mutate(image_urls = ifelse(image_urls != "", strsplit(image_urls, ",\\s*"),"paste link here")) %>%
        dplyr::mutate(youtube_ids = ifelse(youtube_ids != "", strsplit(youtube_ids, ",\\s*"),"paste ID here")) %>%
        dplyr::mutate(cell_conditions = ifelse(cell_conditions != "", strsplit(cell_conditions, ",\\s*"),"")) %>%
        dplyr::mutate(cell_types = ifelse(cell_types != "", strsplit(cell_types, ",\\s*"),""))
      md.fn <- paste(makeFilename(meta[t,"title"]),"md",sep = ".") 
      con = file(file.path("../_trials",md.fn), "w")
      write("---", con)
      yaml::write_yaml(md,con)
      write("---", con)
      close.connection(con)
    }
    #LABS MD
    dir.create("../_labs", showWarnings = FALSE)
    cat.list <- unique(meta$lab)
    for(ct in cat.list){
      ct.fn <- file.path("../_labs",
                         paste(makeFilename(cleanName(ct)),"md",sep = "."))
      if(!file.exists(ct.fn)){
        md <- data.frame(name=cleanName(ct), description=ct)
        con = file(ct.fn, "w")
        write("---", con)
        yaml::write_yaml(md,con)
        write("---", con)
        close.connection(con)
      }
    }
    #INSTITUTIONS MD
    dir.create("../_institutions", showWarnings = FALSE)
    cat.list <- unique(meta$institution)
    for(ct in cat.list){
      ct.fn <- file.path("../_institutions",
                         paste(makeFilename(cleanName(ct)),"md",sep = "."))
      if(!file.exists(ct.fn)){
        md <- data.frame(name=cleanName(ct), description=ct)
        con = file(ct.fn, "w")
        write("---", con)
        yaml::write_yaml(md,con)
        write("---", con)
        close.connection(con)
      }
    }
    #INSTRUMENTS MD
    dir.create("../_instruments", showWarnings = FALSE)
    cat.list <- unique(meta$instrument)
    for(ct in cat.list){
      ct.fn <- file.path("../_instruments",
                         paste(makeFilename(cleanName(ct)),"md",sep = "."))
      if(!file.exists(ct.fn)){
        md <- data.frame(name=cleanName(ct), description=ct)
        con = file(ct.fn, "w")
        write("---", con)
        yaml::write_yaml(md,con)
        write("---", con)
        close.connection(con)
      }
    }
  }
}

################################################################################
## FUNCTIONS ##
###############

cleanName <- function(old){
  new <- gsub(","," -",old)
  return(new)
}

makeFilename <- function(old){
  new <- gsub(" ","_",old)
  return(new)
}

makeSubmissionFormat2 <- function(fn){
  meta <- read.table(fn, sep = "\t", header = T, stringsAsFactors = F)
  meta2 <- as.data.frame(
    meta %>%
      dplyr::select(-c(file,well,cell_count_2)) %>%
      dplyr::mutate(description=cleanName(description)) %>%
      dplyr::group_by(title,description,lab,institution,date,image_url,youtube_ids,row,col,time,frequency,plate,magnification,instrument,cell_type_1,cell_count_1,cell_type_2,cell_condition) %>%
      dplyr::summarise(channels = paste(sort(channel),collapse=", ")) %>%
      dplyr::group_by(title,description,lab,institution,date,image_url,youtube_ids,channels,row,col,frequency,plate,magnification,instrument,cell_type_1,cell_count_1,cell_type_2,cell_condition) %>%
      dplyr::summarise(total_time = last(time), image_count=n()) %>%
      dplyr::rename(image_urls =image_url) %>%
      dplyr::mutate(cell_types=paste(c(cleanName(cell_type_1),cleanName(cell_type_2)), collapse = ", ")) %>%
      dplyr::rename(cell_count = cell_count_1) %>%
      dplyr::rename(cell_conditions = cell_condition) 
  )
  meta2$cell_type_1 <- NULL
  meta2$cell_type_2 <- NULL
  write.table(meta2, "metadata_submission_v2.tsv", sep = "\t", col.names = T, row.names = F)
}
