import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)



with st.sidebar:
  st.title("App Filter Options (Lookup)")
  
  ee_Algorithms = st.expander(label='ee.Algorithms')
  with ee_Algorithms:
    st.markdown(""" 
    * FMask.fillMinima()
    * FMask.matchClouds()
    * GeometryConstructors.BBox()
    * GeometryConstructors.LineString()
    * GeometryConstructors.LinearRing()
    * GeometryConstructors.MultiGeometry()
    * GeometryConstructors.MultiLineString()
    * GeometryConstructors.MultiPoint()
    * GeometryConstructors.MultiPolygon()
    * GeometryConstructors.Point()
    * GeometryConstructors.Polygon()
    * GeometryConstructors.Rectangle()
    * Image.Segmentation.GMeans()
    * Image.Segmentation.KMeans()
    * Image.Segmentation.SNIC()
    * Image.Segmentation.seedGrid()
    * Landsat.TOA()
    * Landsat.calibratedRadiance()
    * Landsat.pathRowLimit()
    * Landsat.simpleCloudScore()
    * Landsat.simpleComposite()
    * Sentinel2.CDI()
    * TemporalSegmentation.Ccdc()
    * TemporalSegmentation.Ewmacd()
    * TemporalSegmentation.LandTrendr()
    * TemporalSegmentation.LandTrendrFit()
    * TemporalSegmentation.StructuralChangeBreakpoints()
    * TemporalSegmentation.VCT()
    * TemporalSegmentation.Verdet()
    * CannyEdgeDetector()
    * Collection(features)
    * CrossCorrelation()
    * Date()
    * Describe()
    * Dictionary()
    * Feature()
    * HillShadow()
    * HoughTransform()
    * If()
    * IsEqual()
    * ObjectType()
    * Proj()
    * ProjectionTransform()
    * String()
    * Terrain()
    """)

  ui_Chart = st.expander(label='ui.Chart')
  with ui_Chart:
    st.markdown(""" 
    * array.values()
    * feature.byFeature()
    * feature.byProperty()
    * feature.groups()
    * feature.histogram()
    * image.byClass()
    * image.byRegion()
    * image.doySeries()
    * image.doySeriesByRegion()
    * image.doySeriesByYear()
    * image.histogram()
    * image.regions()
    * image.series()
    * image.seriesByRegion()
    """)
                    
  ee_Classifier = st.expander(label='ee.Classifier')
  with ee_Classifier:
        st.markdown(""" 
        * amnhMaxent()
        * decisionTree()
        * decisionTreeEnsemble() 
        * libsvm()
        * minimumDistance()
        * smileCart()
        * smileGradientTreeBoost()
        * smileNaiveBayes()
        * smileRandomForest()
        * spectralRegion()
        """)
        
  ee_Clusterer = st.expander(label='ee.Clusterer')
  with ee_Clusterer:
        st.markdown("""     
        * wekaCascadeKMeans()
        * wekaCobweb()
        * wekaKMeans()
        * wekaLVQ()
        * wekaXMeans()
        """)
        
  ee_ImageCollection = st.expander(label='ee.ImageCollection')
  with ee_ImageCollection:
        st.markdown("""     
        * COPERNICUS/S2
        * COPERNICUS/S2_SR
        * COPERNICUS/S1_GRD
        * COPERNICUS/S2_HARMONIZED
        * ECMWF/ERA5_LAND/MONTHLY
        * ESA/WorldCover/v100
        * FAO/WAPOR/2/L1_AETI_D
        * FAO/WAPOR/2/L1_NPP_D
        * FIRMS
        * JAXA/GPM_L3/GSMaP/v6/operational
        * JRC/GSW1_3/MonthlyHistory
        * LANDSAT/LC08/C01/T1_SR
        * LANDSAT/LC08/C02/T1_L2
        * LANDSAT/LC09/C02/T1_L2
        * LANDSAT/LE07/C01/T1_SR
        * LANDSAT/LT05/C01/T1_SR
        * NASA_USDA/HSL/SMAP10KM_soil_moisture
        * projects/sat-io/open-datasets/FABDEM
        * projects/sat-io/open-datasets/WSF/WSF_2019
        * UCSB-CHG/CHIRPS/DAILY
        * UCSB-CHG/CHIRPS/PENTAD
        """)
        
  ee_Join = st.expander(label='ee.Join')
  with ee_Join:
        st.markdown("""     
        * inner()
        * inverted()
        * saveAll()
        * saveBest()
        * saveFirst()
        * simple()""")
  ee_Kernel = st.expander(label='ee.Kernel')
  with ee_Kernel:
        st.markdown("""     
        * chebyshev()
        * circle()
        * compass()
        * cross()
        * diamond()
        * euclidean()
        * fixed()
        * gaussian()
        * kirsch()
        * laplacian4()
        * laplacian8()
        * manhattan()
        * octagon()
        * plus()
        * prewitt()
        * rectangle()
        * roberts()
        * sobel()
        * square()""")
  ee_Reducer = st.expander(label='ee.Reducer')
  with ee_Reducer:
        st.markdown("""     
        * allNonZero()
        * and()
        * anyNonZero()
        * autoHistogram()
        * bitwiseAnd()
        * bitwiseOr()
        * centeredCovariance()
        * circularMean()
        * circularStddev()
        * circularVariance()
        * count()
        * countDistinct()
        * countDistinctNonNull()
        * countEvery()
        * countRuns()
        * covariance()
        * first()
        * firstNonNull()
        * fixed2DHistogram()
        * fixedHistogram()
        * frequencyHistogram()
        * geometricMedian()
        * histogram()
        * intervalMean()
        * kendallsCorrelation()
        * kurtosis()
        * last()
        * lastNonNull()
        * linearFit()
        * linearRegression()
        * max()
        * mean()
        * median()
        * min()
        * minMax()
        * mode()
        * or()
        * pearsonsCorrelation()
        * percentile()
        * product()
        * ridgeRegression()
        * robustLinearRegression()
        * sampleStdDev()
        * sampleVariance()
        * sensSlope()
        * skew()
        * spearmansCorrelation()
        * stdDev()
        * sum()
        * toCollection()
        * toList()
        * variance()""")
  ee_Terrain = st.expander(label='ee.Terrain')
  with ee_Terrain:
        st.markdown("""     
        * aspect()
        * fillMinima()
        * hillShadow()
        * hillshade()
        * products()
        * slope()""")
  



st.title("Earth Engine App Filter")
st.markdown("""
      ##### Creator: Philipp Gärtner  |  Last Update: 23/11/2022
      """)
st.markdown("""
    [![Follow](https://img.shields.io/twitter/follow/Mixed_Pixels?style=social)](https://twitter.com/Mixed_Pixels) [![Mastodon Follow](https://img.shields.io/mastodon/follow/109312882727067778?domain=https%3A%2F%2Fmapstodon.space%2F&style=social)](https://mapstodon.space/@Mixed_Pixels)  [![Coffee for Philipp](https://img.shields.io/badge/Coffee%20for%20Philipp--yellow.svg?logo=buy-me-a-coffee&logoColor=orange&style=social)](https://www.buymeacoffee.com/0l94rzR)
    """)
      
st.write(
    """[Earth Engine Apps](https://developers.google.com/earth-engine/guides/apps) are dynamic, shareable user interfaces for Earth Engine analyses. The [ee-appshot](https://github.com/samapriya/ee-appshot) repository creates a weekly snapshot of available Earth Engine Apps and provides their URL’s and script source codes.

This streamlit app provides an object and method filter which allows the users to quickly get to the app functionality they are interested in.
    """)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df


df = pd.read_csv(
    "ee_df.csv"
)


st.dataframe(filter_dataframe(df))
