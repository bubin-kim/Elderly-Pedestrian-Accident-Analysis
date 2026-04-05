library(tidyverse)

all_accidents <- read_csv("/Users/bbin/Documents/SL/acc_data_2024/tsc_export_sago_file_v.csv")  # 데이터 불러오기

daejeon_clean <- all_accidents %>%
  filter(substr(as.character(bjd_cd), 1, 2) == "30") %>%  # 대전시만 필터링 하기 위해 30으로 시작되는 법정동 코드를 찾고
  select(
    acc_ym,
    acc_ymd,
    wea_sta_cd,
    rd_typ_cd,
    acc_typ_cd,
    bjd_cd,
    acc_grd_cd
  )  # 필요한 컬럼만 선택

str(daejeon_clean)

write.csv(daejeon_clean, "daejeon_accidents.csv", row.names = FALSE)
