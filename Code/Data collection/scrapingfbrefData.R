install.packages("devtools")
devtools::install_github("JaseZiv/worldfootballR")
devtools::install_github("JaseZiv/worldfootballR", force = TRUE)


library(worldfootballR)
packageVersion('worldfootballR')

install.packages("dplyr")
library(dplyr)

remotes::install_github('JaseZiv/worldfootballR', force = TRUE)

install.packages("rvest")
library(rvest)

Sys.setenv(
  CHROMOTE_CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"
)

Sys.getenv("CHROMOTE_CHROME")

stat_types <- list("standard",
                    "shooting",
                    "passing",
                    "passing_types",
                    "gca",
                    "defense",
                    "possession",
                    "playing_time",
                    "misc",
                    "keepers",
                    "keepers_adv")



for(i in stat_types){
  
  df <- fb_big5_advanced_season_stats(season_end_year= c(2021:2026),
                                      stat_type= i,
                                      team_or_player= "player")
  dplyr::glimpse(df)
  
  View(df)
  write.csv(df, sprintf("big5_player_%s.csv", i))
  
}

for(i in stat_types){
  
  df <- fb_big5_advanced_season_stats(season_end_year= c(2021:2026),
                                      stat_type= i,
                                      team_or_player= "team")
  dplyr::glimpse(df)
  
  
  write.csv(df, sprintf("big5_team_%s.csv", i))
  
}

#   country = c("ENG", "ESP")
#   gender = c("M", "F")
#   season_end_year = 2020:2021,
#   tier = c("1st", "2nd"),
#   non_dom_league_url = NA,
#   stat_type = "shooting",
#   team_or_player = "player"
