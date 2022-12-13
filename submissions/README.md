# How to submit your image collection

### There are two ways to submit your image collection to the Occident database and web portal.

1. Use [our simple Google Form](https://forms.gle/bSr2BMC8rJm1Yq278). The form allows you to submit an image collection for a single trial and well. Reuse the form to submit multiple collections.
2. Learn about our tab-delimited file format (TSV) and provide metadata on multiple trials and multiple wells (one per row) using a template of required column headings (see below).

## Template TSV instructions
A [template TSV file](TEMPLATE.tsv) is included in this folder. Each of the columns is described in detail below. Generate a TSV with your image collection metadata using the template and submit it to this [submissions folder](https://github.com/bee-hive/occident/edit/main/submissions) using GitHub.

### Required columns
- `lab` -- The name of the lab responsible for the image data collection.
- `institution` -- The name of the home institution for the responsible lab.
- `title` -- A title for the image collection. ***MUST BE UNIQUE***
- `description` -- An overall description for the image collection.
- `instrument` -- The instrument that collected the image data, e.g., "Incucyte"
- `magnification` -- The magnification setting for the images, e.g., "4x"
- `channels` -- A comma-separated list of channels collected, e.g., "phase, red"
- `plate_type` -- The platform used in the data collection, e.g., "96 well"
- `row` -- The row designation of the well for this image collection, e.g., "B"
- `column` -- The column designation of the well for this image collection, e.g., "2"
- `total_time` -- The total ellapsed time for the image collection, formatted as ##d##h##m, e.g., "03d08h12m"
- `frequency` -- The time interval between images, e.g., "4m"
- `image_count` -- The total number of images in this collection, e.g., "1324"
- `date` -- The start date of the image collection, formatted as MM/DD/YY, e.g., 08/19/22

### Optional columns
- `cell_types` -- A comma-separated list of cell types, e.g., "T Cells, A375"
- `cell_count` -- The total number of cells in the well, e.g., "1500"
- `cell_conditions` -- A comma-separated list of xperimental conditions for the cells, e.g., "RASA2 KO, Drug X47"
- `comments` -- Additional text relevant to the images collected for this well and trial.
- `image_urls` -- A comma-separated list of URLs to download or access the images.
- `youtube_ids` -- A comma-separated list of YouTube IDs for movies of this image collection.


