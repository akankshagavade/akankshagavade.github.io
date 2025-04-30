# Final report


## Summary 

This project aims to determine whether the sentiment in 10-K filings contains value-relevant information for stock returns. Specifically, it investigates whether a document’s positive or negative tone correlates with better or worse stock performance. To conduct this study, we will use textual analysis and sentiment analysis techniques to extract sentiment scores from 10-K filings. These scores will be calculated by applying sentiment dictionaries to the text, thereby identifying positive and negative word frequencies. We take this analysis a step beyond general sentiment by incorporating contextual sentiment, which measures the tone of discussions around specific topics within the filings. The three topics selected are Environment, Manufacturing, and Technology, each assigned both a positive and negative sentiment score using a particular dictionary. 

The primary goal is to see whether sentiment within these topic areas has any predictive power over stock returns. The research follows a cross-sectional event study framework, comparing sentiment measures to stock performance around 10-K release dates. Expected outputs include scatterplots, regression analyses, and heatmaps, all aimed at uncovering patterns between textual sentiment and market reactions.



## Sample

The sample we used is the s&p 500 for the 2022 year. We download the 10-Ks for 2022 for 498 of these firms using sec Edgar downloader. This download can take about 30 minutes.

We fetch the list of all S&P 500 companies from a specific Wikipedia page, saves it as a CSV, and then loads it into a pandas DataFrame. The goal is to ensure that the 2022 list of S&P 500 companies is used, avoiding potential issues with outdated tickers.

Then after checking if a ZIP file of 10-K filings exists, we create a loop which loops through the S&P 500 companies' CIKs and downloads the latest 10-K filings for each company from the SEC EDGAR database. It then removes any previously downloaded .txt files to keep only the .html files. We handles potential errors by pausing (sleep(.2)) if the SEC limits the requests.

All these downloads account for 2.7 GB on your hard drive. Our code compresses the folder containing the 10-K filings into a ZIP file, and reduces its size from 2GB to 150MB.  Finally, we delete the original folder to allow easy access to th e files.

## Building return variables

**1** Our data is sourced from CRSP (the Center for Research on Securities Pricing). After importing and reviewing the file, we find that it contains approximately 2.58 million rows, each corresponding to a ticker/symbol/company in the S&P 500. The dataset includes dates and the respective returns for each day. We use this information to determine the filing date return and calculate cumulative returns over the periods [0,2] and [3,10] days.

**2**
```python
            crsp_returns['p1'] = crsp_returns.groupby('ticker')['date'].rank()
```
Each firm has multiple stock return observations over time. We assign a rank (p1) to each trading date within each firm, creating a sequence of dates. This helps us track trading days relative to the 10-K filing date.

**3**
```python
            crsp_returns['p1'] = crsp_returns['p1'].fillna(0).astype(int)
```
If there are missing values (NaNs), they are replaced with 0 to avoid errors. The rank is then converted to an integer for proper indexing.

**4**
```python
            crsp_returns['temp'] = crsp_returns['p1'] * (crsp_returns['date'] >= crsp_returns['Filing date'])
```
Next we created a temporary variable (temp) that stores 0 for dates before the filing date and the ranked trading day value (p1) for dates on or after the filing date. This helps locate the closest trading date to the 10-K event.

**5**
```python
            mask = crsp_returns['temp'] == 0
            crsp_returns.loc[mask, 'temp'] = 1000
            crsp_returns['p2'] = crsp_returns.groupby('ticker')['temp'].transform('min')
```
Any trading date before the filing date (where temp == 0) is replaced with 1000. This ensures that we only focus on dates at or after the filing as we need to find the minimum of the indexes to correctly identify the row of observation we need,.

**6**
```python
            crsp_returns['p2'] = crsp_returns.groupby('ticker')['temp'].transform('min')
```
We take the minimum value of temp for each firm. Since pre-filing dates were set to 1000, this selects the first available minimum trading day on or after the 10-K filing date.

**7**
```python
            crsp_returns['p2'] = crsp_returns.groupby('ticker')['temp'].transform('min')
```
We take the minimum value of temp for each firm. Since pre-filing dates were set to 1000, this selects the first available minimum trading day on or after the 10-K filing date.

**8**
```python
            crsp_returns.eval('trading_days_since_event = p1 - p2', inplace=True)
```
We take the minimum value of temp for each firm. Since pre-filing dates were set to 1000, this selects the first available minimum trading day on or after the 10-K filing date.


**For example,** each trading day is assigned a sequential number (p1).
So, March 1 is p1 = 1, March 2 is p1 = 2, March 3 is p1 = 3, and so on.
Before the filing date, we set temp = 1000 as a placeholder for those days.
The first p1 value on or after the filing date is assigned as p2 (which is the minimum temp).
Let's say March 3 is the filing date, the minimum p2 is set to 3.
So we can then compute the Trading Days Since Filing (event) by doing (p1 - p2):

This gives the number of trading days relative to the filing date.
March 3, the filing date, is Day 0.
March 4 is 1 day after filing, March 5 is 2 days after, and so on.
March 1 and 2 are before the filing date, so they have negative values (-2, -1).
This method allows us to track stock return observations based on how many days they occur before or after the 10-K filing.



| Date      | p1 | Filing Date | temp  | p2 (Min) | trading_days_since_event (p1 - p2) |
|-----------|------------------|-------------|------|--------------|----------------------------------|
| March 1   | 1                | March 3     | 1000 | 3            | -2                               |
| March 2   | 2                | March 3     | 1000 | 3            | -1                               |
| March 3   | 3                | March 3     | 3    | 3            | 0                                |
| March 4   | 4                | March 3     | 4    | 3            | 1                                |
| March 5   | 5                | March 3     | 5    | 3            | 2                                |
| March 8   | 6                | March 3     | 6    | 3            | 3                                |

- `p1`: The trading rank assigned to each date.
- `temp`: Placeholder for dates before the filing.
- `p2`: The earliest available filing date (`p1` on or after the event).
- `trading_days_since_event`: The difference `p1 - p2`, measuring how far each date is from the filing.

**9**
Compute Cumulative Returns in Different Windows

To compute the returns for day 0 (day of filing event):

**10**
To compute cumulative returns for the 2 windows: 

```python

            def compute_returns(x):
                if x.loc[x['trading_days_since_event'].between(0, 2), 'ret'].notna().sum() > 0:
                    return pd.Series({
                        'Cumulative Return [0,2]': (1 + x.loc[x['trading_days_since_event'].between(0, 2), 'ret']).prod() - 1,
                        'Cumulative Return [3,10]': (1 + x.loc[x['trading_days_since_event'].between(3, 10), 'ret']).prod() - 1
                    })
                else:
                    return None
```  
We group the data by firm (ticker/symbol) and apply this function. If there are non-missing return values in the [0,2] trading days window:
Compute cumulative return over the [0,2] trading days window by using this formula: Cumulative Return = (1 + Daily Return_1) * (1 + Daily Return_2) * ... * (1 + Daily Return_N) or  Σ (1 + Daily Return_i) where i = [0, n]. 

We compute cumulative return over the [3,10] trading days window using the same method but change the .between function from between(0, 2) to between(3, 10).
Otherwise, return None to exclude firms without valid return data.

**11**
```python
            res = crsp_returns.groupby('ticker').apply(compute_returns).dropna().reset_index()
            result = pd.DataFrame(res)
```
We apply the function to each firm's data. The rows with missing values (dropna()) are removed.
Finally, the results are stored in a final DataFrame (result) with cumulative return measures.



## Building sentiment variables

In this section, we discuss how the code processes 10-K filings, extracts text data, and computes sentiment scores for two dictionaries and three contextual topics:


**1**
```python
            with ZipFile('10k_files/10k_files.zip','r') as zipfolder:
```
We load and open the 10-K files that we downloaded before.

**2**
```python
            for index, row in sp500.iterrows():  
                firm_folder = f"sec-edgar-filings/{str(row['CIK']).zfill(10)}/10-K/*/*.html"
                possible_files = fnmatch.filter(file_list, firm_folder)
                if len(possible_files) == 0: 
                    continue
                fpath = possible_files[0]
```
Iterate through each firm's data in the S&P 500 dataset. Then we locate the 10-K HTML file for each firm by using its CIK.


**3**
```python
            with zipfolder.open(fpath) as report_file:
                html = report_file.read().decode(encoding="utf-8")
                soup = BeautifulSoup(html, features='lxml-xml')
```
We extract the HTML from the file and convert it to usable text format for analysis.

**4**
```python
            for div in soup.find_all("div", {'style': 'display:none'}):
                div.decompose()
            document = soup.text.lower()
            document = re.sub(r'\W', ' ', document)  # Remove punctuation
            document = re.sub(r'\s+', ' ', document)  # Normalize whitespace
            doc_length = len(document.split())
```
This code cleans the HTML file. It  converts all tests to lowercase, making the analysis with dictionary much easier. It removes punctuations and extra spaces. Finally it counts the total number of words in the document aand stores it in the variable 'doc_length'.


**5**

Now that we have a cleaned document, we can begin sentiment analysis.
```python
            bhr_neg_regex = r'\b(' + '|'.join(BHR_negative) + r')\b'
            neg_hits = len(re.findall(bhr_neg_regex, document))
            sentiment_score_BHR_negative = neg_hits / doc_length
            sp500.at[index,'BHR Negative'] = sentiment_score_BHR_negative
```
This is the beginning of sentiment analysis.Starting with the BHR dictionary, we use the regex function to count occurrences of negative words from the BHR dictionary.
We calculate: Negative sentiment score = Number of Negative Words/Total Words in Document

```python
            bhr_pro_regex = r'\b(' + '|'.join(BHR_positive) + r')\b'
            
            pro_hits = len(re.findall(bhr_pro_regex, document))
    
            sentiment_score_BHR_positive = pro_hits/doc_length
            sp500.at[index,'BHR Positive']  = sentiment_score_BHR_positive

            lm_pro_regex = r'\b(' + '|'.join(LM_positive) + r')\b'
            
            pro_hits_lm = len(re.findall(lm_pro_regex, document))
    
            sentiment_score_LM_positive = pro_hits_lm/doc_length
            sp500.at[index,'LM Positive']  = sentiment_score_LM_positive

            lm_neg_regex = r'\b(' + '|'.join(LM_negative) + r')\b'
            
            neg_hits_lm = len(re.findall(lm_neg_regex, document))
    
            sentiment_score_LM_negative = neg_hits_lm/doc_length
            sp500.at[index,'LM  Negative']  = sentiment_score_LM_negative
```
We repeat these steps for BHR positive, LM negative and LM positive.



**6**

Then we move on to contextual analysis for our three topics: technology, manufacturing, environment. For each topic, it calculates the positive sentiment score and negative sentiment score.
```python
sentiment_score_tech_positive = NEAR_finder(tech_dict, BHR_positive, document)[0] / doc_length
sp500.at[index,'Technology_positive'] = sentiment_score_tech_positive

```
We use the NEAR_finder function that count how often topic1 (in this case, technology dictionary) is near topic2 (BHR positive sentiment) in a document.
```python
            sentiment_score_tech_negative = NEAR_finder(tech_dict,BHR_negative,document)[0]/doc_length
            sp500.at[index,'Technology_negative']  = sentiment_score_tech_negative
            
            sentiment_score_mfg_positive = NEAR_finder(mfg_dict,BHR_positive,document)[0]/doc_length
            sp500.at[index,'Manufacturing_positive']  = sentiment_score_mfg_positive
            
            sentiment_score_mfg_negative = NEAR_finder(mfg_dict,BHR_negative,document)[0]/doc_length
            sp500.at[index,'Manufacturing_negative']  = sentiment_score_mfg_negative
            
            sentiment_score_env_positive = NEAR_finder(env_dict,BHR_positive,document)[0]/doc_length
            sp500.at[index,'Environment_positive']  = sentiment_score_env_positive
            
            sentiment_score_env_negative = NEAR_finder(env_dict,BHR_negative,document)[0]/doc_length
            sp500.at[index,'Environment_negative']  = sentiment_score_env_negative
```

We repeat this process with the negative BHR dictionary. And then we repeat it for the other 2 topics, and save it to a dataframe using: 

```python
            results.append({'CIK': row['CIK'],'Doc Length': doc_length, 'BHR Negative': sentiment_score_BHR_negative,'BHR Positive': sentiment_score_BHR_positive, 
                                        'LM Positive': sentiment_score_LM_positive,'LM Negative': sentiment_score_LM_negative,
                                        'Tech positive': sentiment_score_tech_positive, 'Tech negative': sentiment_score_tech_negative,
                                        'Env positive': sentiment_score_env_positive, 'Env negative': sentiment_score_env_negative,
                                       'Mfg positive': sentiment_score_mfg_positive, 'Mfg negative': sentiment_score_mfg_negative})
            sentiment_df = pd.DataFrame(results)
```





#### Data check point

| **Dictionary** | **Word Count** |
| ----------- | ----------- |
| BHR_negative| 94 |
| BHR_positive| 75 |
| LM_negative| 2345 |
| LM_positive| 347 |



## Regex setup
I set up the distance to 10, while I tried a few other numbers before. This medium length distance covers almost half a sentence on average and is a good indicator of how relevant the positive or negative htis are to our sentiment analysis. We also set partial = false because some words fo my dictionary could be partially included in other unrelated words, and we did not wish to take the risk.


## Contextual sentiment

For the contextual sentiment analysis, the topics I selected were Manufacturing & Supply Chain, Technology, and Environment. They are highly relevant risk factors in 10-K filings, which can influence firm operations, financial performance, firm reputation and investor sentiment.

#### Environment

Climate-related disclosures in 10-K filings have increased significantly due to new laws, regulatory pressures, ESG mandates, and sustainability commitments. Companies now focus on addressing their carbon footprint, regulatory compliance, and exposure to climate risks, which influence their financial outlook and investor perceptions. Terms like "superfund," "remediation," and "liability act" are typically used in regulatory discussions about ESG implementation. Meanwhile, terms such as "greenhouse gas," "emissions," "carbon," "footprint," and "energy" are commonly used to describe a company's efforts toward environmental conservation. I chose this topic because of my interest in understanding how businesses balance sustainability goals with financial performance, and how environmental factors shape investor perceptions and corporate strategies.

Some excerpts picked up by near_FINDER include: _'sustainability driven two pillar growth strategy that includes expansion','growing portion of our business involves clean hydrogen carbon', 'climate change sustainability and environmental issues has also led to increased', 'solid and hazardous wastes and remediation of contamination'_

Industries like pharmaceuticals and food & beverages tend to speak positively about environmental topics due to strict regulations on health and safety compliance. Companies such as Coca-Cola, Abbott Pharmaceuticals, Mead Johnson, Hershey, and Biogen have high ESG indexes and are recognized among the "greenest" firms, reflecting their commitment to sustainability and regulatory adherence.


#### Manufacturing & Supply Chain

Companies address supply chain resilience, logistics disruptions, and operational risks in their filings due to challenges in global sourcing, cost fluctuations, and geopolitical risks. Sentiment analysis can capture concerns about supply disruptions, inventory management, and supplier relationships. Terms such as "plant," "property," "equipment," "infrastructure," "facilities," and "factories" can reveal insights into manufacturing capabilities from the 10-K. Additionally, I included words like "obsolescence" and "lead time," which suggest issues related to inventory management and supply chain dynamics. This topic is personally interesting to me as it related to my major (industrial & systems engineering) and it explores how companies manage their operational efficiency and thereby, stock returns.

Some excerpts picked up by near_FINDER include: _'manufacturing processes to make them more efficient and to optimize performance', 'continue to invest heavily in marketing research and development new manufacturing facilities', 'manufacturing capacity with the demand of the marketplace resulting in a strong',, 'plant investments any of which may negatively impact our financial results'_

Industries heavily reliant on mass manufacturing, such as consumer goods and automotive sectors, tend to discuss manufacturing and supply chain topics positively. Capital expenditures (CapEx) on plants, property, and equipment are key financial and economic indicators of their success. These factors can signal confidence or weakness to investors, ultimately influencing stock prices.

#### Technology

10-K filings highlight the use of advanced technology in areas like product design and manufacturing. This topic aims to explore a company's investment in technology, with sentiment analysis revealing how firms adapt to innovation and leverage new tech for efficiency to maintain competitiveness. Terms such as "technical infrastructure," "cybersecurity," "information technology," "machine learning," and "artificial intelligence" capture popular discussions about how companies are upgrading their technology and investing in various areas, such as product innovation or other strategic initiatives. I find this topic personally engaging because it sheds light on how technology influences both business growth and competitive advantage in an ever-evolving market.

Some excerpts picked up by near_FINDER include: _'technology enables us to deliver the highest levels of product performance' , 'technologies may become obsolete and replaced by other market alternatives performance'_

Companies deeply rooted in the tech industry, such as Alphabet, Meta, and NVIDIA, often speak positively about this sentiment as it relates to their innovative products, services, and future R&D initiatives.


## Summary stats of the final analysis sample 

![](images/sentiment.png)

These are the summary statistics for our sentiment analysis. At most, only about 3.2% of the firms' documents indicated positive sentiment, with a significant disparity between the minimum and maximum values. It will be interesting to explore the correlations between these sentiment scores and returns, as discussed further in the report.

![](images/sum.png)

These summary statistics indicate that the average returns across the three return measures are very low, with a significant standard deviation and considerable disparity between the minimum and maximum values. On average, after the first week of 10-K filings, the cumulative returns appear negligible, showing little change in return values. However, they can fluctuate between as low as -0.28 and as high as 0.33.

Currently, we have not noticed any caveats in our data so we go ahead with our analysis and results.

## Results

### Correlation table using heatmap

![](images/heatmap.png) 

### Insights from the heat maps: 

The heatmap suggests weak overall relationships, as most correlation values are close to zero for ret_0 but change at the third return meaasure.

- The strongest relationships appear in the environment sentiment, implying that this sector's sentiment is more influential on stock returns than the other two sentiments. Environment_negative has the highest positive correlation (0.278) with Cumulative Return [3,10], followed by Environment positive (0.257) with Cumulative Return [3,10]
  
- Manufacturing_negative also has a notable positive correlation (0.149) with Cumulative Return [3,10] and Manufacturing positive has slightly lwoe coreelation (0.125) for the same window.

- Technology has near-zero correlations with returns, indicating that technology-related discussions in filings might not have a major predictive impact.
  
- LM Negative and LM Positive show consistently negative correlations across all return measures, implying that positive sentiment in these categories may be linked to lower returns. LM Negative has the strongest negative correlation (-0.099 for Cumulative Return [3,10]), suggesting that positive LM sentiment may be a negative return predictor.

- BHR positive and negative have weak positive correlations. BHR positive is consistently positive, but clustered around zero. BHR negative changes signs at the third return measure so we speculate that the market may have been overbought during the first 2 days, causing a decrease in returns for days 3-10.


### Scatter plots [10x3] 
### Insights from the scatter plots: 
- Most scatter plots show weak linear relationships and a straight flat regression line, confirming the low correlations seen in the heatmap above.
- Regression lines in most plots are nearly flat, which indicates the minimal predictive power of these sentiment measures on returns. 
- Data points are  scattered in all plots, meaning there is high variability and little systematic relationship between sentiment and returns.
  
For the environment-related topic, 
- Environment_positive vs. Cumulative Return [3,10] shows a slightly steeper regression line, aligning with its higher correlation (r = +0.26) in the heatmap.
- The data clusters around zero in all plots, showing that extreme sentiment values are rare. Outliers are present but do not strongly influence regression trends or significantly change the overall patterns

For the technology-related topic, 
- Technology_positive vs. Cumulative Return [0,2] shows the highest correlation in that topic (r = +0.08).
- The data clusters between 0.02 and 0.03  in all plots and extreme sentiment values are rare. 

For the manufacturing-related topic, 
- manufacturing_negative vs. Cumulative Return [3,10] shows the highest correlation in that topic (r = +0.15).

LM Positive has a slight downward trend across the first couple return measures, and LM negative has the hgihest correlation in [3,10] window, as seen on the heatmap.
BHR sentiment is clustered aorund the middle of the graph, with no notable correlations.


Most of these correlations are not high enough to make a sound judgement regarding my initial hypothesese. The show no pattern or trend, suggesting that market reactions to these sentiments are highly unpredictable.

![](images/scat1.png)
![](images/scat2.png)

### Regression correlation lines

Most regression lines are relatively flat, indicating that sentiment measures have weak or no strong linear relationships with stock returns.

The shaded regions which are the confidence intervals are quite large, meaning there's high uncertainty in these estimates. This suggests that sentiment variables alone may not be reliable predictors of returns.

There is also a difference in the way sentiments affect returns in the short-term (ret_0) and long term (Cumulative Return [3,10]). In the ret_0 graph (first one), the lines are more tightly packed, meaning sentiment has less impact on very short-term returns. In contrast, for Cumulative Return [3,10], some lines diverge more than before, indicating sentiment could play a larger role in long-term returns.


![](images/regr.png)

## Discussion Topic 1

BHR sentiments (both positive and negative) are positively correlated with returns, while LM sentiments (both positive and negative) are negatively correlated. When BHR sentiment increases (whether positive or negative), same-day market returns (ret_0) also tend to increase. When LM sentiment increases, same-day market returns tend to decrease. This suggests that the market reacts differently to the two different sentiments.

My hypothesis was that negative sentiment would lower returns and positive sentiment would raise them. However, BHR Negative Sentiment (blue) correlates positively with returns, especially in the short term, possibly due to a risk acknowledgment premium. On the other hand, BHR Positive Sentiment (orange) shows a weak positive correlation initially but turns negative for Cumulative Return [3,10], suggesting that initial optimism may lead to an overbought market and a subsequent reversal in returns.


LM Positive’s (green bars) negative correlation is the strongest among all, while BHR Negative (blue bars) has the highest positive correlation. The effect of LM Negative increased over the three return measures, whereas that of LM positive decreased.  This is a little counterintuitive because it suggests that if the LM sentiment is optimisitic, it might encourage the market to get overbought, thus leading to lower returns. Whereas is the opposite maybe true for the BHR sentiment, as in if the BHR sentiment is overly pessimistic, investors might take it as a sign to buy the dip, and drive up returns. 

This indicates that excessive positivity in LM sentiment could be a contrarian indicator, while negative sentiment in BHR sources might signal a short-term buying opportunity. However, overall, the correlation is too close to zero to make an accurate judegment regarding this hypothesis. 




![](images/q1.png)

## Discussion Topic 2

Upon studying Table 3 of the Garcia et al. study, and comparing it with my own results, I can observe both agreements and discrepancies in the patterns. In both studies, LM Positive shows a negative correlation with returns and BHR (ML) Positive shows a positive correlation with returns. Garcia's report also states that these results were statistically significant. 

But a discrepency arisses when comparing LM Negative and  BHR (ML) negative. My study's results show a stronger negative correlation for LM Negative and ML Negative than what Garcia et al. report. And additionally, Garcia's study indicates that these results were not statistically significant.

There are various reasons for this disparity. 

- Difference in sample size and scope of study: The Garcia et al. study includes 76,922 observations across many years (1995-2020). They included all 10-K filings that qualified certain requirememnts, like being able to match to the CRSP database and considering stocks listed on the NYSE, Amex, or NASDAQ. Since they conducted a regression analysis, they limited their observations to cinclude only the filings with available regression variables (size, book-to-market, share turnover, pre-filing period three factor alpha, filing period excess return, and Nasdaq dummy). In contrast, my data was much smaller, as it included only approx 500 10-Ks from the S&P 500.

- The paper mentioned that it is also standard practice for this type of empirical study to keep the event study structure and add both time (quarter-year) and industry fixed effects so they incorporated these fixed effects in their study. My analysis did not include these steps, which may account for differences in the results.
  
- Garcia et al. used both unigrams and bigrams in their analysis, whereas my dictionaries was limited to unigrams only. This difference could lead to variations in the results.

They included a lot of these additional controls so because these additional data points help smooth outstatistical noise and increase generalizability of their study. More variables and controls can help account for confounding factors and ensure that their results aren't due to other industry factors, so it enhances the credibility of their work.

Their extensive efforts were aiming to produce more reliable results.


## Discussion Topic 3

My original hypothesis was that, among the three contextual sentiments, technology would have the greatest impact on returns, followed by manufacturing, with environmental sentiment having the least effect. Given the rise of AI and ML, I expected every company to speak positively about these topics. Since a significant portion of the S&P 500 consists of consumer goods companies across various sectors, I also anticipated that manufacturing would be a prominent topic. However, the results pleasantly surprised me.

Out of the three contextual sentiments, none of the sentiments show a "different enough" figure for the return on day 0. 

That said, for cumulative returns for [3, 8] window, the environment sentiment stands out with more notable correlations that look slightly different from zero, followed by the manufacturing sentiment, while the technology sentiment present near-zero correlation (refer to the heatmap). This suggests that investor reactions to environment-related news impact returns profoundly in the longer window.

From an economic standpoint, environment-related laws and activities greatly affect firm reputation and consumer trust. 
- Firms that proactively adopt green technologies and eco-friendly practices can enhance their brand reputation, attract ESG-focused investors, and gain a market edge over competitors.
  
- Non-compliance with environmental regulations can result in hefty fines, legal battles, and reputational damage, all of which can negatively affect stock prices and investor confidence.

- Companies also allocate significant resources to comply with environmental laws, such as emission caps, waste disposal regulations, and carbon taxes. This increases operational costs but can also drive efficiency improvements.

Manufacturing sentiment has the second-strongest correlation with the returns. Manufacturing_negative presents +0.149 with Cumulative Return [3,10]) and Manufacturing_positive with +0.125 with Cumulative Return [3,10]. 

- Manufacturing sentiment matters because it’s a core driver of economic output, GDP, and employment. When industrial production expands, it can lead to higher corporate earnings and better stock market performance. A slowdown, on the other hand, can signal economic trouble.

- It’s also tied to supply chain efficiency, which plays a big role in predicting costs, inflation trends, and overall economic stability. Disruptions in manufacturing (due to shortages, delays, rising raw material prices or unforeseen events like pandemics) can have a domino effect on multiple industries, making it a key sentiment measure for investors.

- Since manufacturing involves a huge workforce, its growth means more job creation and higher wages. This, in turn, boosts people’s purchasing power, which can stimulate demand and positively affect economic activity and stock markets, at a domestic and international level.

- Manufacturing sentiment reflects capital expenditure (CAPEX) trends, which can indicate confidence. An increase in CAPEX usually means companies are optimistic about future growth and expansion, but a cut in spending could suggest uncertainty or cost-cutting strategies.


In conclusion, my initial hypothesis was incorrect, likely due to an error in execution. My measurement approach may have been flawed, as I suspect that my technology-related dictionary was too narrow and limited. To avoid false positives, I deliberately kept the dictionary short, because I reason that most companies frequently mention technology, so I aimed for more specificity. In the future, I would run multiple iterations of this project using different dictionary variations and compare the results.


## Discussion Topic 4 

**Focus on how the “ML sentiment” variables (positive and negative) are related to the different return measures. Is there a difference in the sign and magnitude? Speculate on why or why not.** 

The bar chart below how BHR Positive and BHR Negative sentiment variables correlate with the three return measures (ret_0, Cumulative Return [0,2], Cumulative Return [3,10]).

My inital hypothesis was that higher negative sentiments can signal risks and damages, which should lower stock returns. Along the same lines, a higher positive sentiment should singal towards a higher stock return. 

BHR Negative Sentiment (blue bars) is consistently positive across all return measures. Largest correlation for Cumulative Return [0,2], indicating that negative sentiment about BHR is associated with a slight increase in returns. However, the data contradicts my hypothesis, showing a positive correlation between negative BHR sentiment and returns, particularly in the short term. This may have something to do with risk acknowledgement premium, where investors actually reward companies for acknowledging risks that are captured by the BHR Negative sentiment.

BHR Positive Sentiment (orange bars) also shows positive correlation but is weaker than BHR Negative.
The correlation remains almost zero across return measures but goes negative for Cumulative Return [3,10]. This is where the sign changes and my initial hypothesis does not hold true anymore. 

I speculate this may be because investors initially respond positively to optimistic language. Over the following week [day 3 to 10], if many investors decide to invest based on this overly optimistic sentiment, it can cause the market to being overbought, thereby causing a reversal in returns.


![](images/q1.png)
