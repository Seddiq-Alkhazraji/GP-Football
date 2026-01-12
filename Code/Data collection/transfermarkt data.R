# install.packages("devtools")
devtools::install_github("JaseZiv/worldfootballR")
remotes::install_github('JaseZiv/worldfootballR', force = TRUE)

library(worldfootballR)
library(dplyr)
library(rvest)

countries <- c("England", "Spain", "France", "Italy", "Germany")

years <- c(2025)

team_urls <- c()

for (i in 1:length(years)) {
  urls <- tm_league_team_urls(country_name = countries[i], start_year = years[i])
  team_urls <- c(team_urls, urls)
}


#---------------------------------------------------------------------------


eur_xfers <- tm_team_transfers(team_url = team_urls, transfer_window = "all")

write.csv(eur_xfers, "big5_transfers.csv")

#---------------------------------------------------------------------------

library(worldfootballR)
library(dplyr)
library(purrr)

# --- 1. SETUP ---
countries <- c("England", "Spain", "France", "Italy", "Germany")
years <- c(2020:2024) 

# --- 2. GET URLS (CORRECTED LOGIC) ---
# We use nested loops (or expand.grid) to ensure we get ALL countries for ALL years
team_urls <- c()

message("Getting Team URLs...")
for (cntry in countries) {
  for (yr in years) {
    message(paste("Fetching URLs for:", cntry, yr))
    
    # Wrap in tryCatch in case a specific league page fails
    tryCatch({
      urls <- tm_league_team_urls(country_name = cntry, start_year = yr)
      team_urls <- c(team_urls, urls)
    }, error = function(e) {
      message(paste("  Error fetching URLs for", cntry, yr))
    })
  }
}

team_urls <- unique(team_urls) # Remove duplicates just in case
message(paste("Total teams to scrape:", length(team_urls)))

# --- 3. SCRAPE TRANSFERS (SAFE METHOD) ---
# Instead of passing all URLs at once, we define a "Safe" function
# This prevents one bad team page from crashing the entire loop

safe_scrape_transfers <- function(url) {
  message(paste("Scraping:", url))
  
  result <- tryCatch({
    # The actual scraping function
    tm_team_transfers(team_url = url, transfer_window = "all")
  }, error = function(e) {
    message(paste("  ❌ FAILED:", url))
    message(paste("  Reason:", e$message))
    return(NULL) # Return NULL if it fails, so we can keep going
  })
  
  return(result)
}

# --- 4. EXECUTE ---
# map_df will loop through every URL, run the safe function, and combine results
# This might take a while!
all_transfers <- team_urls %>% 
  purrr::map_df(safe_scrape_transfers)

# --- 5. CHECK & SAVE ---
print(head(all_transfers))
write.csv(all_transfers, "all_transfers_safe.csv", row.names = FALSE)
message("Done!")

#---------------------------------------------------------------------------
# let's get the transfer balances for the 2020/21 Bundesliga season
team_balances <- tm_team_transfer_balances(country_name = "Germany", start_year = 2020)

#----- Can do it for a single league: -----#
bundesliga_valuations <- tm_player_market_values(country_name = "Germany",
                                               start_year = 2021)
dplyr::glimpse(a_league_valuations)


#--------------------------------------------------------------------
#----- Can also do it for multiple leagues: -----#
leagues_urls <- list("https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1", 
                     "https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES1", 
                     "https://www.transfermarkt.com/ligue-1/startseite/wettbewerb/FR1", 
                     "https://www.transfermarkt.com/serie-a/startseite/wettbewerb/IT1", 
                     "https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/L1")

big_5_valuations <- c()

big_5_valuations <- tm_player_market_values(country_name = c("England", "Spain", "France", "Italy", "Germany"),
                                           start_year = c(2020:2024))

for (i in leagues_urls) {
  vl <- tm_player_market_values(country_name =c("England", "Spain", "France", "Italy", "Germany"),
                                start_year = c(2020:2024),
                                league_url = i)
  big_5_valuations <- c(big_5_valuations, vl)
}

write.csv(big_5_valuations, "big5_market_value.csv")



#---------------------------------------------------------------------------



team_urls <- c()

for (i in 1:length(countries)) {
  urls <- tm_league_team_urls(country_name = countries[i], start_year = 2024)
  team_urls <- c(team_urls, urls)
}



tm_team_player_urls <- function(team_url) {
  main_url <- "https://www.transfermarkt.com"
  
  team_page <- tryCatch(
    xml2::read_html(team_url),
    error = function(e) {
      message(paste("Failed to read:", team_url))
      return(NULL)
    }
  )
  
  if (is.null(team_page)) {
    return(NULL)
  }
  
  player_urls <- team_page %>%
    rvest::html_nodes("#yw1 td.posrela .hauptlink a") %>%
    rvest::html_attr("href") %>%
    unique() %>%
    paste0(main_url, .)
  
  return(player_urls)
}



#----- for multiple players: -----#
# # can make use of a tm helper function:
players_urls <- tm_team_player_urls(team_url = team_urls[1])
#for (i in 1:length(team_urls)) {
#  urls <- tm_team_player_urls(team_url = team_urls[i])

#  players_urls <- c(players_urls, urls)
#}



showConnections(all = TRUE)
for (con in showConnections(all = TRUE)[, "description"]) {
  try(close(getConnection(as.integer(con))), silent = TRUE)
}
closeAllConnections()



players_urls <- c()
all_players_urls <- c()
failed_urls <- c()
still_failed <- c()
for (i in seq_along(team_urls)) {
  urls <- tm_team_player_urls(team_url = team_urls[i])
  
  if (is.null(urls)) {
    failed_urls <- c(failed_urls, team_urls[i])
  } else {
    all_players_urls <- c(all_players_urls, urls)
  }
  
  Sys.sleep(runif(1, 1.5, 3.5))
}

failed_urls <- final_failed_urls
# Try again on failed URLs
for (url in failed_urls) {
  urls <- tm_team_player_urls(team_url = url)
  
  if (is.null(urls)) {
    still_failed <- c(still_failed, url)
  }else{
    all_players_urls <- c(all_players_urls, urls)
  }
  
  Sys.sleep(runif(1, 2, 4))
}



final_failed_urls <- c()
max_retries <- 3
retry_count <- 1

while (length(still_failed) > 0 && retry_count <= max_retries) {
  message(paste("Retry round", retry_count, "with", length(still_failed), "URLs..."))
  current_failures <- c()
  
  for (url in still_failed) {
    urls <- tm_team_player_urls(url)
    
    if (is.null(urls)) {
      current_failures <- c(current_failures, url)
    } else {
      all_players_urls <- c(all_players_urls, urls)
    }
    
    Sys.sleep(runif(1, 2, 4))
  }
  
  still_failed <- current_failures 
  #failed_urls <- current_failures  # Only retry what's still failing
  retry_count <- retry_count + 1
}

final_failed_urls <- still_failed


write.csv(all_players_urls, "big5_players_urls.csv")
write.csv(final_failed_urls, "big5_players_failed_urls.csv")


players_urls_df <- data.frame(all_players_urls)
players_failed_urls_df <- data.frame(final_failed_urls)


#---------------------------------------------------------------------------
for (i in 1:length(team_urls)) {
  print(i)
  print(team_urls[i])
}
all_players_urls <- read.csv("C:/Users/ASUS/Documents/Big 5 leagues data/big5_players_urls.csv")

players_bios <- data.frame()

failed_bios <- c()

for (i in seq_along(all_players_urls)){
  bio <- tm_player_bio(player_urls = all_players_urls[i])

      if (is.null(bio)) {
        failed_bios <- c(failed_bios, all_players_urls[i])      }
      else {
        players_bios <- rbind(players_bios, bio)      }


  #saveRDS(players_bios, "players_bios_partial.rds")
  #players_bios <- rbind(players_bios, tm_player_bio(player_urls = all_players_urls[i]))
  print(all_players_urls[i])
  Sys.sleep(runif(1, 5, 10))  # Random delay between 5 to 10 seconds
  
}
# # then pass all those URLs to the tm_player_bio
#players_bios <- tm_player_bio(player_urls = all_players_urls)

write.csv(players_bios, "big5_player_bios.csv")

players_bios <- readRDS("players_bios_partial.rds")


#---------------------------------------------------------------------------


#----- for multiple players: -----#
# # then pass all those URLs to the tm_player_injury_history
player_injuries <- tm_player_injury_history(player_urls = all_players_urls)

write.csv(player_injuries, "big5_player_injury.csv")


