from DeepTreeAttention.generators import create_training_shp
import pytest
import geopandas as gpd
from shapely.geometry import Point

@pytest.fixture()
def testdata():
    path = "data/raw/test_with_uid.csv"
    shp = create_training_shp.test_split(path)
    
    assert not shp.empty
    
    return shp

def test_test_split():
    path = "data/raw/test_with_uid.csv"
    shp = create_training_shp.test_split(path)
    assert not shp.empty
    assert all([x in ["siteID","plotID","elevation","domainID","individualID","taxonID","itcEasting","itcNorthing","geometry"] for x in shp.columns])
    
    print("There are {} records for {} species for {} sites in train".format(
        shp.shape[0],
        len(shp.taxonID.unique()),
        len(shp.siteID.unique())
    ))
    
def test_train_split(testdata):
    path = "data/raw/latest_full_veg_structure.csv"
    shp = create_training_shp.train_split(path, testdata.individualID, testdata.taxonID, debug=True)
    assert not shp.empty
    assert all([x in ["siteID","plotID","height","elevation","domainID","individualID","taxonID","itcEasting","itcNorthing","geometry"] for x in shp.columns])
    
    print("There are {} records for {} species for {} sites in test".format(
        shp.shape[0],
        len(shp.taxonID.unique()),
        len(shp.siteID.unique())
    ))    
        
def test_filter_CHM():
    lookup_glob = "data/raw/*CHM*"
    
    #Create fake data, within one good point and one bad point by looking at the CHM -> true value is 12.43
    good_point = {"geometry":Point(320224.5, 4881567.7), "height": 12, "plotID":"BART_000"}
    bad_point = {"geometry":Point(320224.5, 4881567.7), "height": 0, "plotID": "BART_000"}
    traindata = gpd.GeoDataFrame([good_point, bad_point])
    filtered_data = create_training_shp.filter_CHM(traindata, lookup_glob=lookup_glob)
    assert filtered_data.shape[0] == 1