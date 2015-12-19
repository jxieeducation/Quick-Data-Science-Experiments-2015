Basically an analysis of mobile app store data

mongoexport --jsonArray --host mobiledata.bigdatacorp.com.br -u GitHubCrawlerUser -p g22LrJvULU5B -d MobileAppsData -c PlayStore_2015_11 -fields=Developer,Url,Name,Reviews --port 21766 -o android_reviews.dump

