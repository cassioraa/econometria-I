import pandas as pd

def aggregate_to_quarterly(data, method='average'):
    """
    Aggregates monthly economic data to quarterly frequency using different methods.

    Args:
        data (dict or pandas.DataFrame): The input data containing monthly values. If a dictionary is provided,
            keys represent the monthly dates in string format (e.g., 'YYYY-MM'), and values are the corresponding
            data for each month. If a DataFrame is provided, it should have a datetime index representing the
            monthly dates, and columns representing the variables.
        method (str, optional): The aggregation method to use. Defaults to 'average'. Possible values are:
            - 'average': Takes the average of each quarter.
            - 'sum': Sums the values for each quarter.
            - 'last': Takes the last value of each quarter.

    Returns:
        pandas.DataFrame: The aggregated data with quarterly frequency, where the index represents the
            quarterly periods and the columns represent the variables.

    Raises:
        ValueError: If an invalid aggregation method is specified.

    Examples:
        # Sample monthly data
        monthly_data = {
            '2023-01': [10, 20, 30],
            '2023-02': [15, 25, 35],
            '2023-03': [12, 22, 32],
            '2023-04': [18, 28, 38],
            '2023-05': [11, 21, 31],
            '2023-06': [13, 23, 33]
        }

        # Aggregate the data using different methods
        aggregated_average = aggregate_to_quarterly(monthly_data, method='average')
        aggregated_sum = aggregate_to_quarterly(monthly_data, method='sum')
        aggregated_last = aggregate_to_quarterly(monthly_data, method='last')

        print("Average aggregation:\n", aggregated_average)
        print("\nSum aggregation:\n", aggregated_sum)
        print("\nLast value aggregation:\n", aggregated_last)
    """
    # Convert the monthly data to a DataFrame with a datetime index
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)

    # Resample the data to quarterly frequency using the specified method
    if method == 'average':
        # Take the average of each quarter
        aggregated_data = df.resample('Q').mean()
    elif method == 'sum':
        # Sum the values for each quarter
        aggregated_data = df.resample('Q').sum()
    elif method == 'last':
        # Take the last value of each quarter
        aggregated_data = df.resample('Q').last()
    else:
        raise ValueError('Invalid aggregation method')

    # Convert the datetime index back to quarterly periods
    aggregated_data.index = aggregated_data.index.to_period('Q')

    return aggregated_data
