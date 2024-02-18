from seasons import Seasons
from holidays import Holidays
from functools import partial
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')


class TemperatureDataAnalyzer:
    """Class to analyze temperature data."""

    def __init__(self, file_path):
        """
        Initialize the TemperatureDataAnalyzer object.

        Parameters:
        - file_path (str): The file path to the data.
        """
        self.file_path = file_path
        self.data: pd.DataFrame = None
        self.read_data()

    def __str__(self):
        return f"{self.summary_statistics()}"

    def read_data(self):
        """Read data from the provided file path."""
        try:
            xls = pd.ExcelFile(self.file_path)
            self.data = pd.read_excel(xls, 'data')
        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")

    def summary_statistics(self, filter_func=None):
        """Display summary statistics of the data."""
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return
        data = self.data
        if filter_func is not None:
            data = data[filter_func(data)]
        summary = data.describe()
        return summary

    def get_pure_data(self):
        """Get the raw data."""
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        return self.data

    def get_data(
            self,
            time,
            base=None,
            filter_func=None) -> pd.DataFrame | None:
        """
        Get filtered data based on time and optionally a base column.

        Parameters:
        - time (str): The time column to group by.
        - base (str, optional): The column to select as base. Defaults to None.
        - filter_func (function, optional): A filter function to apply to the data. Defaults to None.

        Returns:
        - pd.DataFrame | None: The filtered data, grouped by time, optionally with base column.
        """
        data = self.get_pure_data()
        if data is None:
            print("No data to analyze. Please read the data first.")
            return None

        if base == 'SRA':
            data = data.dropna(subset=['SRA'])
        if filter_func is not None:
            data = data[filter_func(data)]
        if base is None:
            return data.groupby(time)
        return data.groupby(time)[base]

    def plot(self, title, xlabel, ylabel, *data):
        """
        Plot the data.

        Parameters:
        - title (str): The title of the plot.
        - xlabel (str): The label for the x-axis.
        - ylabel (str): The label for the y-axis.
        - *data (pd.Series): Series of data to plot.
        """
        if not data:
            print("No data to plot. Please provide valid data.")
            return
        for i in data:
            i.plot(label=i.name)
        plt.title(title)
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def plot_monthly_avg_temperature(self, filter_func=None):
        """
        Plot monthly average temperature.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.
        """
        # Retrieve monthly average temperature data
        monthly_avg_temp_data = self.get_data(
            'měsíc', 'T-AVG', filter_func=filter_func)
        if monthly_avg_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return
        # Calculate monthly average temperature
        monthly_avg_temp = monthly_avg_temp_data.mean()
        # Plot the data
        self.plot(
            "Monthly Average Temperature",
            "Month",
            "Temperature (°C)",
            monthly_avg_temp)

    def plot_monthly_max_temperature(self, filter_func=None):
        """
        Plot monthly maximum temperature.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.
        """
        # Retrieve monthly maximum temperature data
        monthly_max_temp_data = self.get_data(
            'měsíc', 'TMA', filter_func=filter_func)
        if monthly_max_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate monthly maximum temperature
        monthly_max_temp = monthly_max_temp_data.max()
        # Plot the data
        self.plot(
            "Monthly Max Temperature",
            "Month",
            "Temperature (°C)",
            monthly_max_temp)

    def plot_monthly_min_temperature(self, filter_func=None):
        """
        Plot monthly minimum temperature.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.
        """
        # Retrieve monthly minimum temperature data
        monthly_min_temp_data = self.get_data(
            'měsíc', 'TMI', filter_func=filter_func)
        if monthly_min_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate monthly minimum temperature
        monthly_min_temp = monthly_min_temp_data.min()
        # Plot the data
        self.plot(
            "Monthly Min Temperature",
            "Month",
            "Temperature (°C)",
            monthly_min_temp)

    def plot_monthly_all_temperature(self, filter_func=None):
        """
        Plot monthly average, maximum, and minimum temperatures.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.
        """
        # Partial function to retrieve temperature data
        temperature = partial(
            self.get_data,
            time='měsíc',
            filter_func=filter_func)
        # Retrieve monthly average temperature data
        monthly_avg_temp_data = temperature(base='T-AVG')
        if monthly_avg_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate and name monthly average temperature
        monthly_avg_temp = monthly_avg_temp_data.mean()
        monthly_avg_temp.name = "Average Temperature"
        # Calculate and name monthly maximum temperature
        monthly_max_temp = temperature(base='TMA').max()
        monthly_max_temp.name = "Max Temperature"
        # Calculate and name monthly minimum temperature
        monthly_min_temp = temperature(base='TMI').min()
        monthly_min_temp.name = "Min Temperature"
        # Plot the data
        self.plot(
            "Monthly Temperature",
            "Month",
            "Temperature (°C)",
            monthly_avg_temp,
            monthly_max_temp,
            monthly_min_temp)

    def plot_yearly_all_temperature(self, filter_func=None):
        """
        Plot yearly average, maximum, and minimum temperatures.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.
        """
        # Partial function to retrieve temperature data
        temperature = partial(
            self.get_data,
            time='rok',
            filter_func=filter_func)
        # Retrieve yearly average temperature data
        monthly_avg_temp_data = temperature(base='T-AVG')
        if monthly_avg_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate and name yearly average temperature
        monthly_avg_temp: pd.DataFrame = monthly_avg_temp_data.mean()
        monthly_avg_temp.name = "Average Temperature"
        # Calculate and name yearly maximum temperature
        monthly_max_temp = temperature(base='TMA').max()
        monthly_max_temp.name = "Max Temperature"
        # Calculate and name yearly minimum temperature
        monthly_min_temp = temperature(base='TMI').min()
        monthly_min_temp.name = "Min Temperature"
        # Plot the data
        self.plot(
            "Yearly Temperature",
            "Year",
            "Temperature (°C)",
            monthly_avg_temp,
            monthly_max_temp,
            monthly_min_temp)

    def plot_monthly_avg_rainfall(self, filter_func=None):
        """
        Plot monthly average rainfall.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.
        """
        # Retrieve monthly average rainfall data
        monthly_avg_rainfall_data = self.get_data(
            'měsíc', 'SRA', filter_func=filter_func)
        if monthly_avg_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate monthly average rainfall
        monthly_avg_rainfall = monthly_avg_rainfall_data.mean()
        # Plot the data
        self.plot(
            "Monthly Average Rainfall",
            "Month",
            "Rainfall (mm)",
            monthly_avg_rainfall)

    def plot_monthly_max_rainfall(self, filter_func=None):
        """
        Plot monthly maximum rainfall.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.
        """
        # Retrieve monthly maximum rainfall data
        monthly_max_rainfall_data = self.get_data(
            'měsíc', 'SRA', filter_func=filter_func)
        if monthly_max_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate monthly maximum rainfall
        monthly_max_rainfall = monthly_max_rainfall_data.max()
        # Plot the data
        self.plot(
            "Monthly Max Rainfall",
            "Month",
            "Rainfall (mm)",
            monthly_max_rainfall)

    def plot_monthly_min_rainfall(self, filter_func=None):
        """
        Plot monthly minimum rainfall.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.
        """
        # Retrieve monthly minimum rainfall data
        monthly_min_rainfall_data = self.get_data(
            'měsíc', 'SRA', filter_func=filter_func)
        if monthly_min_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate monthly minimum rainfall
        monthly_min_rainfall = monthly_min_rainfall_data.min()
        # Plot the data
        self.plot(
            "Monthly Min Rainfall",
            "Month",
            "Rainfall (mm)",
            monthly_min_rainfall)

    def plot_monthly_all_rainfall(self, filter_func=None):
        """
        Plot the monthly average, maximum, and minimum rainfall.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.
        """
        # Retrieve rainfall data for each month
        rainfall = self.get_data('měsíc', 'SRA', filter_func=filter_func)
        if rainfall is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate monthly average, maximum, and minimum rainfall
        monthly_avg_rainfall = rainfall.mean()
        monthly_avg_rainfall.name = "Average Rainfall"
        monthly_max_rainfall = rainfall.max()
        monthly_max_rainfall.name = "Max Rainfall"
        monthly_min_rainfall = rainfall.min()
        monthly_min_rainfall.name = "Min Rainfall"

        # Plot the data
        self.plot(
            "Monthly Rainfall",
            "Month",
            "Rainfall (mm)",
            monthly_avg_rainfall,
            monthly_max_rainfall,
            monthly_min_rainfall)

    def plot_day_of_month_avg_temperature(self, filter_func=None):
        """
        Plot the daily average temperature.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.
        """
        # Retrieve daily average temperature data
        daily_avg_temp_data = self.get_data(
            ['měsíc', 'den'], 'T-AVG', filter_func=filter_func)
        if daily_avg_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate daily average temperature
        daily_avg_temp = daily_avg_temp_data.mean()

        # Plot the data
        self.plot(
            "Daily Average Temperature",
            "Day",
            "Temperature (°C)",
            daily_avg_temp)

    def plot_day_of_month_max_temperature(self, filter_func=None):
        """
        Plots the daily maximum temperature for each day of the month.

        Args:
            filter_func (function, optional): A function to filter the data before plotting.

        Returns:
            None
        """
        # Get daily maximum temperature data
        daily_max_temp_data = self.get_data(
            ['měsíc', 'den'], 'TMA', filter_func=filter_func)
        if daily_max_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate daily maximum temperature
        daily_max_temp = daily_max_temp_data.max()

        # Plot the data
        self.plot(
            "Daily Max Temperature",
            "Day",
            "Temperature (°C)",
            daily_max_temp)

    def plot_day_of_month_min_temperature(self, filter_func=None):
        """
        Plots the daily minimum temperature for each day of the month.

        Args:
            filter_func (function, optional): A function to filter the data before plotting.

        Returns:
            None
        """
        # Get daily minimum temperature data
        daily_min_temp_data = self.get_data(
            ['měsíc', 'den'], 'TMI', filter_func=filter_func)
        if daily_min_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate daily minimum temperature
        daily_min_temp = daily_min_temp_data.min()

        # Plot the data
        self.plot(
            "Daily Min Temperature",
            "Day",
            "Temperature (°C)",
            daily_min_temp)

    def plot_day_of_month_avg_rainfall(self, filter_func=None):
        """
        Plots the daily average rainfall for each day of the month.

        Args:
            filter_func (function, optional): A function to filter the data before plotting.

        Returns:
            None
        """
        # Get daily average rainfall data
        daily_avg_rainfall_data = self.get_data(
            ['měsíc', 'den'], 'SRA', filter_func=filter_func)
        if daily_avg_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate daily average rainfall
        daily_avg_rainfall = daily_avg_rainfall_data.mean()

        # Plot the data
        self.plot(
            "Daily Average Rainfall",
            "Day",
            "Rainfall (mm)",
            daily_avg_rainfall)

    def plot_day_of_month_max_rainfall(self, filter_func=None):
        """
        Plots the daily maximum rainfall for each day of the month.

        Args:
            filter_func (function, optional): A function to filter the data before plotting.

        Returns:
            None
        """
        # Get daily maximum rainfall data
        daily_max_rainfall_data = self.get_data(
            ['měsíc', 'den'], 'SRA', filter_func=filter_func)
        if daily_max_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate daily maximum rainfall
        daily_max_rainfall = daily_max_rainfall_data.max()

        # Plot the data
        self.plot(
            "Daily Max Rainfall",
            "Day",
            "Rainfall (mm)",
            daily_max_rainfall)

    def plot_day_of_month_min_rainfall(self, filter_func=None):
        """
        Plots the daily minimum rainfall for each day of the month.

        Args:
            filter_func (function, optional): A function to filter the data before plotting.

        Returns:
            None
        """
        # Get daily minimum rainfall data
        daily_min_rainfall_data = self.get_data(
            ['měsíc', 'den'], 'SRA', filter_func=filter_func)
        if daily_min_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate daily minimum rainfall
        daily_min_rainfall = daily_min_rainfall_data.min()

        # Plot the data
        self.plot(
            "Daily Min Rainfall",
            "Day",
            "Rainfall (mm)",
            daily_min_rainfall)

    def plot_rainfall_mean_development_decades(self, filter_func=None):
        """
        Plot the mean development of rainfall aggregated by decades.

        Args:
            filter_func (function, optional): A function to filter the data before analysis. Defaults to None.
        """
        # Retrieve rainfall data
        rainfall = self.get_data('rok', 'SRA', filter_func=filter_func)
        if rainfall is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate mean rainfall for each decade
        rainfall_mean = rainfall.mean()
        rainfall_mean_decades = rainfall_mean.groupby(
            rainfall_mean.index // 10 * 10).mean()

        # Plot the data
        self.plot(
            "Rainfall Mean Development Decades",
            "Decade",
            "Rainfall (mm)",
            rainfall_mean_decades)

    def plot_rainfall_mean_development_years(
            self, number_of_years=1, filter_func=None):
        """
        Plot the mean development of rainfall aggregated by specified number of years.

        Args:
            number_of_years (int, optional): Number of years to aggregate data. Defaults to 1.
            filter_func (function, optional): A function to filter the data before analysis. Defaults to None.
        """
        # Retrieve rainfall data
        rainfall = self.get_data('rok', 'SRA', filter_func=filter_func)
        if rainfall is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate mean rainfall for each specified number of years
        rainfall_mean = rainfall.mean()
        rainfall_mean_years = rainfall_mean.groupby(
            rainfall_mean.index // number_of_years * number_of_years).mean()

        # Plot the data
        self.plot(
            f"Rainfall Mean Development Years (By {number_of_years} years)",
            "Year",
            "Rainfall (mm)",
            rainfall_mean_years)

    def plot_rainfall_mean_development_month(self, month):
        """
        Plot the mean development of rainfall for a specific month.

        Args:
            month (int): The month for which rainfall data is to be plotted.
        """
        # Retrieve rainfall data for the specified month
        rainfall = self.get_data(
            'rok', 'SRA', filter_func=lambda x: x['měsíc'] == month)
        if rainfall is None:
            print("No data to analyze. Please read the data first.")
            return

        # Calculate mean rainfall for the specified month
        rainfall_mean = rainfall.mean()

        # Plot the data
        self.plot(
            f"Rainfall Mean Development For Month {month}",
            "Year",
            "Rainfall (mm)",
            rainfall_mean)

    def __get_outliers(self, column, offset=3, filter_func=None):
        """
        Get outliers for a specific column in the weather data.

        Parameters:
        - column (str): The column name for which outliers are to be identified.
        - offset (float): The number of standard deviations away from the mean to consider as an outlier.
        - filter_func (function): A function to filter the data before identifying outliers.

        Returns:
        - DataFrame: A DataFrame containing the outliers.
        """
        # Retrieve the pure data and drop rows with missing values in the
        # specified column
        data = self.get_pure_data().dropna(subset=[column])

        # Check if there's data to analyze
        if data is None:
            print("No data to analyze. Please read the data first.")
            return

        # Apply any filtering function provided
        if filter_func is not None:
            data = data[filter_func(data)]

        # Compute the z-score for the specified column
        z = np.abs(stats.zscore(data[column]))

        # Return the rows where z-score exceeds the specified offset
        return data[z > offset]

    def get_temperature_outliers(self, offset=3, filter_func=None):
        """
        Get outliers for the average temperature.

        Parameters:
        - offset (float): The number of standard deviations away from the mean to consider as an outlier.
        - filter_func (function): A function to filter the data before identifying outliers.

        Returns:
        - DataFrame: A DataFrame containing the outliers in average temperature.
        """
        return self.__get_outliers('T-AVG', offset, filter_func)

    def get_rainfall_outliers(self, offset=3, filter_func=None):
        """
        Get outliers for rainfall data.

        Parameters:
        - offset (float): The number of standard deviations away from the mean to consider as an outlier.
        - filter_func (function): A function to filter the data before identifying outliers.

        Returns:
        - DataFrame: A DataFrame containing the outliers in rainfall.
        """
        return self.__get_outliers('SRA', offset, filter_func)

    def get_max_temperature_outliers(self, offset=3, filter_func=None):
        """
        Get outliers for the maximum temperature.

        Parameters:
        - offset (float): The number of standard deviations away from the mean to consider as an outlier.
        - filter_func (function): A function to filter the data before identifying outliers.

        Returns:
        - DataFrame: A DataFrame containing the outliers in maximum temperature.
        """
        return self.__get_outliers('TMA', offset, filter_func)

    def get_min_temperature_outliers(self, offset=3, filter_func=None):
        """
        Get outliers for the minimum temperature.

        Parameters:
        - offset (float): The number of standard deviations away from the mean to consider as an outlier.
        - filter_func (function): A function to filter the data before identifying outliers.

        Returns:
        - DataFrame: A DataFrame containing the outliers in minimum temperature.
        """
        return self.__get_outliers('TMI', offset, filter_func)

    def filter_month(self, start_month, end_month=None):
        """
        Filter records based on month(s).

        Args:
            start_month (int): Starting month.
            end_month (int, optional): Ending month. Defaults to None.

        Returns:
            function: Filter function.
        """
        if end_month is None:
            return lambda x: x['měsíc'] == start_month
        if end_month < start_month:
            return lambda x: (
                x['měsíc'] >= start_month) | (
                x['měsíc'] <= end_month)
        return lambda x: (
            x['měsíc'] >= start_month) & (
            x['měsíc'] <= end_month)

    def filter_year(self, start_year, end_year=None):
        """
        Filter records based on year(s).

        Args:
            start_year (int): Starting year.
            end_year (int, optional): Ending year. Defaults to None.

        Returns:
            function: Filter function.
        """
        if end_year is None:
            return lambda x: x['rok'] == start_year
        return lambda x: (x['rok'] >= start_year) & (x['rok'] <= end_year)

    def filter_month_year(
            self,
            start_month,
            start_year,
            end_month=None,
            end_year=None):
        """
        Filter records based on month(s) and year(s).

        Args:
            start_month (int): Starting month.
            start_year (int): Starting year.
            end_month (int, optional): Ending month. Defaults to None.
            end_year (int, optional): Ending year. Defaults to None.

        Returns:
            function: Filter function.
        """
        return lambda x: (
            self.filter_month(
                start_month,
                end_month)(x)) & (
            self.filter_year(
                start_year,
                end_year)(x))

    def filter_season(self, season: Seasons):
        """
        Filter records based on season.

        Args:
            season (Seasons): Season enum.

        Returns:
            function: Filter function.
        """
        match season:
            case Seasons.WINTER:
                return self.filter_month(12, 2)
            case Seasons.SPRING:
                return self.filter_month(3, 5)
            case Seasons.SUMMER:
                return self.filter_month(6, 8)
            case Seasons.AUTUMN:
                return self.filter_month(9, 11)
            case _:
                return None

    def filter_day(self, start_day, end_day=None):
        """
        Filter records based on day(s).

        Args:
            start_day (int): Starting day.
            end_day (int, optional): Ending day. Defaults to None.

        Returns:
            function: Filter function.
        """
        if end_day is None:
            return lambda x: x['den'] == start_day
        if end_day < start_day:
            return lambda x: (x['den'] >= start_day) | (x['den'] <= end_day)
        return lambda x: (x['den'] >= start_day) & (x['den'] <= end_day)

    def filter_date(
            self,
            start_day,
            start_month,
            end_day=None,
            end_month=None):
        """
        Filter records based on date(s).

        Args:
            start_day (int): Starting day.
            start_month (int): Starting month.
            end_day (int, optional): Ending day. Defaults to None.
            end_month (int, optional): Ending month. Defaults to None.

        Returns:
            function: Filter function.
        """
        return lambda x: (
            self.filter_month(
                start_month,
                end_month)(x)) & (
            self.filter_day(
                start_day,
                end_day)(x))

    def filter_holiday(self, holiday: Holidays):
        """
        Filter records based on holiday.

        Args:
            holiday (Holidays): Holiday enum.

        Returns:
            function: Filter function.
        """
        match holiday:
            case Holidays.NEW_YEAR:
                return self.filter_date(1, 1)
            case Holidays.EASTER:
                return self.filter_date(1, 4, 30, 4)
            case Holidays.CHRISTMAS:
                return self.filter_date(24, 12, 26, 12)
            case Holidays.HALLOWEEN:
                return self.filter_date(31, 10)
            case _:
                return None

    def filter_decade(self, start_year, end_year=None):
        """
        Filter records based on decade.

        Args:
            start_year (int): Starting year of the decade.
            end_year (int, optional): Ending year of the decade. Defaults to None.

        Returns:
            function: Filter function.
        """
        if end_year is None:
            return lambda x: (x['rok'] // 10) * 10 == start_year
        return lambda x: (x['rok'] // 10) * \
            10 >= start_year & (x['rok'] // 10) * 10 <= end_year

    def hottest_day_of_year(self, filter_func=None):
        """
        Find the hottest day of the year.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the hottest day of each year.
        """
        data = self.get_pure_data()
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        if filter_func is not None:
            data = data[filter_func(data)]
        idx = data.groupby('rok')['TMA'].idxmax()
        return data.loc[idx]

    def coldest_day_of_year(self, filter_func=None):
        """
        Find the coldest day of the year.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the coldest day of each year.
        """
        data = self.get_pure_data()
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        if filter_func is not None:
            data = data[filter_func(data)]
        idx = data.groupby('rok')['TMI'].idxmin()
        return data.loc[idx]

    def rainiest_day_of_year(self, filter_func=None):
        """
        Find the rainiest day of the year.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the rainiest day of each year.
        """
        data = self.get_pure_data()
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        if filter_func is not None:
            data = data[filter_func(data)]
        idx = data.groupby('rok')['SRA'].idxmax()
        return data.loc[idx]

    def least_rainy_day_of_year(self, filter_func=None):
        """
        Find the least rainy day of the year.

        Args:
            filter_func (function, optional): A function to filter data. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the least rainy day of each year.
        """
        data = self.get_pure_data()
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        if filter_func is not None:
            data = data[filter_func(data)]
        idx = data.groupby('rok')['SRA'].idxmin()
        return data.loc[idx]

    def get_year_max_temperature(self, year):
        """
        Get the maximum temperature for a specific year.

        Args:
            year (int): The year to retrieve maximum temperature data for.

        Returns:
            pandas.DataFrame: DataFrame containing the maximum temperature data for the specified year.
        """
        return self.hottest_day_of_year(self.filter_year(year))

    def get_year_min_temperature(self, year):
        """
        Get the minimum temperature for a specific year.

        Args:
            year (int): The year to retrieve minimum temperature data for.

        Returns:
            pandas.DataFrame: DataFrame containing the minimum temperature data for the specified year.
        """
        return self.coldest_day_of_year(self.filter_year(year))

    def get_year_max_rainfall(self, year):
        """
        Get the maximum rainfall for a specific year.

        Args:
            year (int): The year to retrieve maximum rainfall data for.

        Returns:
            pandas.DataFrame: DataFrame containing the maximum rainfall data for the specified year.
        """
        return self.rainiest_day_of_year(self.filter_year(year))

    def get_highest_rainfall_date(self, filter_func=None):
        """
        Get the date with the highest recorded rainfall.

        Args:
            filter_func (function, optional): A function to filter the data before analysis. Defaults to None.

        Returns:
            tuple: A tuple containing the year, month, day, and rainfall amount for the date with the highest rainfall.
        """
        # Retrieve rainfall data
        data = self.rainiest_day_of_year(filter_func)
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        idx = data['SRA'].idxmax()
        return (
            data['rok'][idx],
            data['měsíc'][idx],
            data['den'][idx],
            data['SRA'][idx])

    def get_lowest_rainfall_date(self, filter_func=None):
        """
        Get the date with the lowest recorded rainfall.

        Args:
            filter_func (function, optional): A function to filter the data before analysis. Defaults to None.

        Returns:
            tuple: A tuple containing the year, month, day, and rainfall amount for the date with the lowest rainfall.
        """
        # Retrieve rainfall data
        data = self.least_rainy_day_of_year(filter_func)
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        idx = data['SRA'].idxmin()
        return (
            data['rok'][idx],
            data['měsíc'][idx],
            data['den'][idx],
            data['SRA'][idx])

    def get_highest_temperature_date(self, filter_func=None):
        """
        Get the date with the highest temperature.

        Args:
            filter_func (function, optional): A function used to filter the data. Defaults to None.

        Returns:
            tuple: A tuple containing the year, month, day, and highest temperature.
        """
        data = self.hottest_day_of_year(filter_func)
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        idx = data['TMA'].idxmax()
        return (
            data['rok'][idx],
            data['měsíc'][idx],
            data['den'][idx],
            data['TMA'][idx])

    def get_lowest_temperature_date(self, filter_func=None):
        """
        Get the date with the lowest temperature.

        Args:
            filter_func (function, optional): A function used to filter the data. Defaults to None.

        Returns:
            tuple: A tuple containing the year, month, day, and lowest temperature.
        """
        data = self.coldest_day_of_year(filter_func)
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        idx = data['TMI'].idxmin()
        return (
            data['rok'][idx],
            data['měsíc'][idx],
            data['den'][idx],
            data['TMI'][idx])

