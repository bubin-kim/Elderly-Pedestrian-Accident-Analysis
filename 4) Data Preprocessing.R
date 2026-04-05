library(tidyverse)

all_accidents <- read_csv("/Users/bbin/Documents/SL/acc_data_2024/tsc_export_sago_file_v.csv")

daejeon_accidents <- all_accidents %>% filter('bjd_cd' == 30) 

daejeon_accidents <- daejeon_accidents %>%
  select('acc_ym', 'acc_ymd', 'wea_sta_cd', 'rd_typ_cd', 'acc_typ_cd', 'bjd_cd', 'acc_grd_cd') 

daejeon_clean <- daejeon_accidents %>%
  select('acc_ym', 'acc_ymd', 'wea_sta_cd', 'rd_typ_cd', 'acc_typ_cd', 'bjd_cd', 'acc_grd_cd')

str(daejeon_clean)

write.csv(daejeon_clean, "daejeon_accidents.csv", row.names = FALSE)
