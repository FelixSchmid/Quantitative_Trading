##################################################################################
### Felix Schmid - Quantitative Trading: Assignment 4
##################################################################################

##################################################################################
### Out-of-sample: 01.01.2018 - 03.08.2020
##################################################################################

from scipy import stats
import statsmodels.tsa.stattools as ts
import math
import numpy as np
import pandas as pd

from quantopian.pipeline.data.morningstar import Fundamentals
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.filters.morningstar import Q1500US, Q500US
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint

##################################################################################
## Helper functions
##################################################################################

def grading_to_numbers(df, col):
    # Helper function that cleans ordinal values to numbers
    df[col] = df[col].astype('object')
    dictionary  = {u'A': 0.1, u'B': 0.3,
                   u'C': 0.7, u'D': 0.9,
                   u'F': 1.0}
    df = df.replace({col: dictionary})
    return df

def find_cointegrated_pairs(data, significance=0.05):
    # Helper function to find a list of cointegrated pairs
    # This function is from https://www.quantopian.com/lectures/introduction-to-pairs-trading
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            # Test for co-integration: Null hypothesis is that
            # there is no co-integration
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

#################################################################################
## Pipeline and initialization
#################################################################################

def make_pipeline():
    universe = Q500US()
    
    market_cap = morningstar.valuation.market_cap.latest.quantiles(5)
    financial_health = morningstar.asset_classification.financial_health_grade.latest
    #total_revenue = Fundamentals.total_revenue.latest
    pe_ratio = Fundamentals.pe_ratio.latest
    #de_ratio = Fundamentals.total_debt_equity_ratio.latest
    revenue_growth = morningstar.operation_ratios.revenue_growth.latest
    # sales to price

    pipe_fundamentals = Pipeline( 
        columns= {
            'Market Cap': market_cap,
            'Financial Health': financial_health,
            #'Total Revenue' : total_revenue,
            'P/E Ratio' : pe_ratio,
            #'D/E Ratio' : de_ratio,
            'Revenue Growth': revenue_growth
        },
        screen=universe
    )
    return pipe_fundamentals
    
def initialize(context):
    attach_pipeline(make_pipeline(), 'pipe_fundamentals')  
    
    # Set up environment
    set_commission(commission.PerShare(cost=0.03, min_trade_cost=None))
    set_slippage(slippage.FixedSlippage(spread=0.00))
    
    # Use a custom function to schedule trades
    schedule_function(func=trade,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(hours=1))
    
    # Initialize model parameters
    context.found_pairs     = False
    context.entry_amount    = 1000
    context.entry_threshold = 1.5
    context.exit_threshold  = 0.0
    context.adf_threshold   = 0.1
    context.lookback        = 252

def handle_data(context, data):
    # This function is required, but we don't do anything here
    pass


#################################################################################
## Model building functions
#################################################################################

def find_pairs(context, data):
    # This functions performs a PCA and Clustering on the Q500.
    # It returns a list of potential pairs that pass a cointegration test.
    
    # get the current date
    #date = str(get_datetime().date())
    
    # Call data from pipeline
    fundamentals = pipeline_output('pipe_fundamentals')
    # During in sample testing at some point the pricing data had NAs for the
    # following stocks. If I had more time, I would investigate it and build the
    # Algorithm more robust.
    # This quick fix, of course could introduce a bias.
    #fundamentals.drop(symbols('PFE','FRX','HNZ'), inplace=True) # dropped during in-sample
    fundamentals.drop(symbols('MON'), inplace=True) # dropped during out-sample testing
    # Apart from dropping MON and run a second time, I did not touch the out-of-sample, 
    # but, of course, that can introduce a bias.
    # In-sample: 01.01.2013 - 12.31.2017
    # Out-of-sample: 01.01.2018 - 03.08.2020
    
    # Transfrom ordinal letters to numbers
    fundamentals = grading_to_numbers(fundamentals,'Financial Health')
    filtered_fundamentals = fundamentals.dropna()
    
    # get historical pricing data for PCA and clustering
    pricing = data.history(fundamentals.index, 
                                 fields="price", 
                                 bar_count=context.lookback, 
                                 frequency="1d")
    
    # Create returns
    pricing_filtered = pricing.T[pricing.T.index.isin(filtered_fundamentals.index)].T
    returns = pricing_filtered.pct_change()
    
    # Droping stocks with missing values
    # We want a limited amount of pairs anyway
    returns = returns.iloc[1:,:].dropna(axis=1)
    
    #PCA
    number_of_pc = 35 
    pca = PCA(n_components=number_of_pc)
    pca.fit(returns)
    
    #Stacking fundamentals with principal components of returns
    X = np.hstack(
        (pca.components_.T,
         # Adding fundamentals to the results of the PCA
         fundamentals['Market Cap'][returns.columns].values[:, np.newaxis],
         fundamentals['Financial Health'][returns.columns].values[:, np.newaxis],
         #fundamentals['Total Revenue'][returns.columns].values[:, np.newaxis],
         fundamentals['P/E Ratio'][returns.columns].values[:, np.newaxis],
         #fundamentals['D/E Ratio'][returns.columns].values[:, np.newaxis],
         fundamentals['Revenue Growth'][returns.columns].values[:, np.newaxis] 
        )
    )
    
    # Normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X) 
    
    # Clustering with DBSCAN
    clf = DBSCAN(eps=1.9, min_samples=4)
    clf.fit(X)
    labels = clf.labels_
    # -1 because DBSCAN does not assign a cluster to all labels,
    # these are marked with '-1'. To count clusters we need to
    # substract 1.
    n_clusters_ = len(set(labels))-1
    print("Number of clustered found: %d" % n_clusters_)
    clustered = clf.labels_
    clustered_series = pd.Series(index=returns.columns, data=clustered.flatten())
    # data points that are not to be assigned to one cluster are marked with -1
    clustered_series = clustered_series[clustered_series != -1]
       
    # This code block is from: 
    # https://www.quantopian.com/posts/pairs-trading-with-machine-learning
    # It orders stocks in cluster and performs cointergration test within
    # the clusters. Pairs is a list with all pairs that passed this test.
    cluster_dict = {}
    counts = clustered_series.value_counts()
    print('stocks per cluster: ')
    print(clustered_series.value_counts())
              
    CLUSTER_SIZE_LIMIT = 9999
    ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]
    for i, which_clust in enumerate(ticker_count_reduced.index):
        tickers = clustered_series[clustered_series == which_clust].index
        score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(pricing[tickers])
        cluster_dict[which_clust] = {}
        cluster_dict[which_clust]['score_matrix'] = score_matrix
        cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
        cluster_dict[which_clust]['pairs'] = pairs
    pairs = []
    for clust in cluster_dict.keys():
        pairs.extend(cluster_dict[clust]['pairs'])
    
    # Print number of pairs
    print('Number of pairs: ' + str(len(pairs)))
    
    # We filter for pairs in a way that each symbol is only contained
    # once in all pairs.
    # That means we do not have a concentration of investment into 
    # one particular company.
    unique_symbols = []
    unique_symbols_pairs = []
    for pair in pairs:
        if pair[0] not in unique_symbols \
            and pair[1] not in unique_symbols:
            unique_symbols_pairs.append(pair)
            unique_symbols.extend(pair)
            
    #################################################
    #####  Testing trading logic with 3 pairs   #####
    #################################################
    #unique_symbols_pairs = unique_symbols_pairs[0:3]
    #print('Testing only 3 pairs is on')


    # Returns the list of pairs that are in one cluster
    # and that are cointegrated. Also, no asset is contained
    # more than once.
    print('Number of pairs with unique assets: ' + str(len(unique_symbols_pairs)))
    return unique_symbols_pairs, unique_symbols
    

def build_model(context, data):
    
    # For the first time that we build the model, we search a list
    # of potential pairs based on the PCA, clustering and cointegration test.
    # Each symbol is only contained once.
    if context.found_pairs == False:
        pairs, symbols = find_pairs(context, data)
        context.number_of_pairs = len(pairs)
        context.pairs_list = pairs
        context.symbols = symbols
        
        # Initializing dummy variables:
        # We only cluster and define pairs with the first run.
        # Therefore, the number of possible traded pairs do not 
        # change from here on.
        context.entry_sign_list = [9999] * len(pairs)
        context.is_traded = [False] * len(pairs)
        
        # set initialization to true
        context.found_pairs = True

   
    # get prices for rebuilding the model
    prices = data.history(context.symbols, 
                          fields="price", 
                          bar_count=context.lookback, 
                          frequency="1d")
    # There is no fill parameter in data.history()
    print(prices.columns[prices.isnull().any()])
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    
    # Initializing lists for the coefficients 
    alpha_list = list()
    beta_list = list()
    spread_std_list = list()
    adf_pvalue_list = list()

    for pair in context.pairs_list:          
        # Here we look back at historical data and build our model

        # Get data
        #prices = history(context.lookback, '1d', 'price', ffill=True)
    
        # run a linear regression on the pair
        beta, alpha, r_value, p_value, std_err = stats.linregress(prices[pair[0]],
                                                              prices[pair[1]])
    
        # use the results of the linear regression to predict the second
        # fund from the first
        predicted = alpha + beta * prices[pair[0]]
    
        # calculate the spread
        spread = prices[pair[1]] - predicted
    
        # Check to see if the spread is stationary. We will compare this
        # value to context.adf_threshold, but generally we are looking for
        # a value of less than 0.05. The lower the value, the higher our 
        # confidence that the pair is cointegrated.
        adf_pvalue = ts.adfuller(spread)[1]
    
        # get the standard deviation of the spread
        spread_std = spread.std()
    
        # store relevant parameters to be used later
        alpha_list.append(alpha)
        beta_list.append(beta)
        spread_std_list.append(spread_std)
        adf_pvalue_list.append(adf_pvalue)
    
    # Safe the list with results of the co-integration tests in context
    # to use access it later on in the trading logic
    context.alpha_list = alpha_list
    context.beta_list = beta_list
    context.spread_std_list = spread_std_list
    context.adf_pvalue_list = adf_pvalue_list    

def return_current(context, data, i):
    # calculate current spread and z-score of spread    
    current_spread = (data[context.pairs_list[i][1]].price - 
                     (context.alpha_list[i] + context.beta_list[i] * data[context.pairs_list[i][0]].price))
    current_z = current_spread / context.spread_std_list[i]
    return current_spread, current_z


#################################################################################
## Trading logic 
## Calls the model building functions
#################################################################################

def trade(context, data):
    # Here's the main trading logic   
    # Trade is called on daily basis

    #########################################################################
    # We continue to rebuild the cointegration models to
    # exit the postion when the potentially new equilibrium is reached.
    # this could produce unexpected returns, but it also might be
    # safer if the relationship of the pair alters substantially 
    # after the trade has been entered.
    #
    # However, PCA and Clustering is only performed once in the beginning!

    build_model(context, data)
    #########################################################################
    
    # iterator to acces safed context lists with variables of each pair
    i = 0    
    for pair in context.pairs_list:
        #print(i)
        #print(pair)
        
        # calculate current relationship of pair
        current_spread, current_z = return_current(context, data, i)
        # check sign of relationship (above or below equilibrium)
        sign = math.copysign(1, current_z)
    
        # time to exit?
        if context.is_traded[i]==True and np.any(sign != context.entry_sign_list[i] or
                                                 abs(current_z) < context.exit_threshold):
            #print('SELL')
            
            # if we get here we were in a trade and the pair has come back
            # to equilibrium.
            order_target_percent(context.pairs_list[i][0], 0)
            order_target_percent(context.pairs_list[i][1], 0)
            context.is_traded[i] = False
    
        # look to enter a trade
        if context.is_traded[i] == False:        
        
            if (context.adf_pvalue_list[i] < context.adf_threshold and     # cointegrated
                abs(current_z) >= context.entry_threshold):        # spread is big enough
                
                #print('BUY')
                # record relationship at start of position
                context.entry_sign_list[i] = sign
            
                # calculate shares to buy based on entry_amount
                # tried to do this with order_target_value() but
                # that would truncate instead of round. This method
                # get's us closer starting values
                shares_pair0 = round(context.entry_amount / data[context.pairs_list[i][0]].price, 0)
                shares_pair1 = round(context.entry_amount / data[context.pairs_list[i][1]].price, 0)
            
                order(context.pairs_list[i][0],      sign * shares_pair0)
                order(context.pairs_list[i][1], -1 * sign * shares_pair1)
                context.is_traded[i] = True
        # iteration
        i = i + 1

    # Records
    record(number_of_pairs = len(context.pairs_list))
    record(number_of_traded_pairs = sum(context.is_traded))
    #record(cash = context.portfolio.cash)