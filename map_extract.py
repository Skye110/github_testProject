import geopandas as gpd

PBF_PATH = "D:/3r_kurs/4-2/map/map_mongolia.pbf"
OUT_SHP  = "D:/3r_kurs/4-2/map/gis_osm_roads_free_1.shp"


print(" Reading .pbf file...")
# read the "lines" layer (roads)
gdf = gpd.read_file(PBF_PATH, layer="lines")

print(" File loaded. Filtering highways...")
# filter only real roads
roads = gdf[gdf["highway"].notna()]
print(f" Total roads: {len(roads)}")

roads.to_file(OUT_SHP)

print("Shapefile written successfully ->", OUT_SHP)
