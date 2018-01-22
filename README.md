# Brian's Data Space

## Project 1: Analyzing NYC MTA subway and demographic data to optimize street team deployment
19 January 2018
### Summary
During the four days of the first week (due to MLK Jr. Day) of Metis' Winter Data Science Bootcamp 2018, we were tasked with optimizing deployment of a street team to raise awareness of a summer gala for Women Tech Women Yes (WTWY) in New York City.

We assumed a modest street team head count, and thus set out to identify the top NYC subway stations which demonstrated both high traffic and favorable demographics.

Our approach was to analyze two sets of data, consisting of:
- MLA Turnstile data from data.ny.gov(https://data.ny.gov/Transportation/Turnstile-Usage-Data-2016/ekwu-khcy); and
- Demographic data, which was an aggregatation of data from:
--  the U.S. Census Bureau(https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml);
--  the Federal Communications Commission(https://www.fcc.gov/general/census-block-conversions-api); and
--  inspired by MuonNeutrino's post on Mapping New York City Census Data(https://www.kaggle.com/muonneutrino/mapping-new-york-city-census-data/data) on Kaggle.com.

### Approach
The complete Jupyter Notebook, including both MTA subway and demographic data analyses, can be found [here](https://github.com/cipher813/McMahon_Metis/blob/master/Project_1/P1_MTA.ipynb).  

#### MTA Approach
Our approach to the MTA Subway Data analysis was as follows:
1. **Import data.** We used full calendar year 2016 turnstile data from [data.ny.gov](https://data.ny.gov/Transportation/Turnstile-Usage-Data-2016/ekwu-khcy).  
2. **Clean data.** Extra spaces were parsed, unnecessary columns were removed, etc.  
3. **Adjust turnstile data for daily increments.** The turnstile data consisted of rolling totals for each individual turnstile.  We thus adjusted by taking the daily incremental ticks of each machine on a daily basis.  
4. **Filter outliers (ie > 99% quantile).** If a turnstile broke down or was reset, the rolling total would be affected.  We adjusted for large irregular values by filtering any number greater than the 99% quantile.  
5. **Sort by traffic.** Traffic aggregated by station was sorted to determine the busiest stations.  Traffic is a metric for general activity, calculated as the sum of entries and exits for each turnstile.  
6. **Plot time series of busiest stations.** The top five stations were further analyzed by plotting the full year time series for each.  

#### MTA Findings
Our results shows that the top MTA subway stations generally had the following characteristics:
- central Manhattan
- multiline subway hubs

The top 20 stations by traffic is as follows:
![alt text](https://github.com/cipher813/McMahon_Metis/blob/master/Project_1/Top20.png "Top 20 Stations")

System traffic:
![alt text](https://github.com/cipher813/McMahon_Metis/blob/master/Project_1/charts/SystemTraffic.png "System Traffic")

As you can see, the top 5 stations consist of:
1. Penn Station
2. 23rd Street
3. Herald Square
4. Times Square
5. Grand Central

With a quick comparison to online data, such as at [web.mta.info](http://web.mta.info/nyct/facts/ffsubway.htm), we are fairly comfortable with these results. However, as this was a project completed from start to finish in less than four days, we have a list of items to follow up on at a later time.  One of these would be to further refine the filtering of outliers from these results.  For instance, there is a spike in March in all of the above charts which we would like to investigate further.  

Diving deeper into these stations:
![alt text](https://github.com/cipher813/McMahon_Metis/blob/master/Project_1/charts/34TH.png "Herald Square 34th Street Daily Traffic, 2016")
![alt text](https://github.com/cipher813/McMahon_Metis/blob/master/Project_1/charts/GRD.png "Grand Central 42nd Street Daily Traffic, 2016")
![alt text](https://github.com/cipher813/McMahon_Metis/blob/master/Project_1/charts/Penn.png "Penn Station 34th Street Daily Traffic, 2016")
![alt text](https://github.com/cipher813/McMahon_Metis/blob/master/Project_1/charts/TSQ.png "Times Square 42nd  Street Daily Traffic, 2016")
![alt text](https://github.com/cipher813/McMahon_Metis/blob/master/Project_1/charts/23RD.png "23rd Street Daily Traffic, 2016")

As you can see, on a full system basis all usage patterns are quite similar and consistent. This is likely due to the fact that these are the top stations by usage and as such capacity is more or less consistently reached on a daily basis.  We can clearly see the weekly dip on the weekends, and a large spike in March which should be investigated further.  

Once we had mapped out the top stations, we then took a dive into the demographic data for each top station to confirm that demographics were favorable for each.  

### Demographic Validation
Using the [American Community Survey data](https://www.kaggle.com/muonneutrino/mapping-new-york-city-census-data/data) as compiled by MuonNeutrino on Kaggle, analyzed demographics of busiest stations.  We then combined census location data from the [U.S. Census Bureau](https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml).
We zoomed in on the highest traffic stations and close proximity to analyze average demographics
utilizing MuonNeutrino's mapping functionality to map demographics.

We determined the following characteristics to make up the target profile for the street teams to focus on as they are deployed in the selected subway stations to raise awareness of the WTWY gala:
1. High income
2. High female population
3. High % use the NYC Subway
4. Business / tech hubs
