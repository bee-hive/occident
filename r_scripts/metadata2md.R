# Process metadata tables into web portal tsv and md files

# Reads any .txt files in same folder, parses content, and then produces
# .md and .tsv files into ../_trials and ../_data, respectively. Also 
# produces unique .md files for  ../_cell-types and ../_conditions.

library(dplyr)
library(yaml)

## GET TXT FILES
meta.files <- list.files(pattern = ".txt")
## PROCESS TXT FILES
for(meta.f in meta.files){
  print(meta.f)
  ## READ TXT
  meta <- read.table(meta.f, sep = "\t", header = T, stringsAsFactors = F)
  meta <- dplyr::mutate(meta,title = makeTitle(file))
  ## PRODUCE FILES
  title.list <- unique(meta$title)
  dir.create("../_data", showWarnings = FALSE)
  dir.create("../_trials", showWarnings = FALSE)
  for(t in title.list){
    meta.t <- meta %>%
      dplyr::filter(title == t) %>%
      dplyr::select(-title)
    #TSV
    tsv.fn <- paste(makeFilename(makeTitle(t)),"tsv",sep = ".")
    write.table(meta.t, file.path("../_data",tsv.fn), sep = "\t", col.names = T, row.names = F)
    #TRIALS MD
    md.fn <- paste(makeFilename(makeTitle(t)),"md",sep = ".")
    md <- data.frame(title=makeTitle(t), 
                     description=t,
                     well = meta.t[1,]$well,
                     row = meta.t[1,]$row,
                     col = meta.t[1,]$col,
                     'total_time' = meta.t[nrow(meta.t),]$time,
                     plate = meta.t[1,]$plate,
                     magnification = meta.t[1,]$magnification,
                     instrument = meta.t[1,]$instrument,
                     'cell_type_1' = cleanName(meta.t[1,]$cell.Type.1),
                     'cell_count_1' = meta.t[1,]$cell.Count.1,
                     'cell_condition_1' = cleanName(meta.t[1,]$cell.Condition.1),
                     'cell_type_2' = cleanName(meta.t[1,]$cell.Type.2),
                     'cell_count_2' = meta.t[1,]$cell.Count.2,
                     'video_url' = "paste link here",
                     'images_url' = "paste link here"
                     )
    con = file(file.path("../_trials",md.fn), "w")
    write("---", con)
    yaml::write_yaml(md,con)
    write("---", con)
    close.connection(con)
  }
  #CELL TYPE MD
  dir.create("../_cell-types", showWarnings = FALSE)
  ct1.list <- unique(meta$cell.Type.1)
  for(ct in ct1.list){
    ct.fn <- paste(makeFilename(cleanName(ct)),"md",sep = ".")
    md <- data.frame(name=cleanName(ct), description=ct)
    con = file(file.path("../_cell-types",ct.fn), "w")
    write("---", con)
    yaml::write_yaml(md,con)
    write("---", con)
    close.connection(con)
  }
  #CONDITION MD
  dir.create("../_conditions", showWarnings = FALSE)
  cond1.list <- unique(meta$cell.Condition.1)
  for(con in cond1.list){
    con.fn <- paste(makeFilename(cleanName(con)),"md",sep = ".")  
    md <- data.frame(name=cleanName(con), description=con)
    con = file(file.path("../_conditions",con.fn), "w")
    write("---", con)
    yaml::write_yaml(md,con)
    write("---", con)
    close.connection(con)
  }
}


################################################################################
## FUNCTIONS ##
###############

makeTitle <- function(old){
  new <- gsub("_\\d{2}d\\d{2}h\\d{2}m\\.tif","",old)
  return(new)
}

cleanName <- function(old){
  new <- gsub(","," -",old)
  return(new)
}

makeFilename <- function(old){
  new <- gsub(" ","_",old)
  return(new)
}
