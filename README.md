# McMahon_Metis

## Project 1: Analyzing NYC MTA subway and demographic data to optimize street team deployment
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
The complete Jupyter Notebook, including both MTA subway and demographic data analyses, can be found here[link].  

Our approach to the MTA Subway Data analysis was as follows:
1. **Import data.** We used full calendar year 2016 turnstile data.  
2. **Clean data.** Extra spaces were parsed, unnecessary columns were removed, etc.  
3. **Adjust turnstile data for daily increments.** The turnstile data consisted of rolling totals for each individual turnstile.  We thus adjusted by taking the daily incremental ticks of each machine on a daily basis.  
4. **Filter outliers (ie > 99% quantile).** If a turnstile broke down or was reset, the rolling total would be affected.  We adjusted for large irregular values by filtering any number greater than the 99% quantile.  
5. **Sort by traffic.** Traffic aggregated by station was sorted to determine the busiest stations.  Traffic is a metric for general activity, calculated as the sum of entries and exits for each turnstile.  
6. **Plot time series of busiest stations.** The top five stations were further analyzed by plotting the full year time series for each.  

#### 


