import shapefile
import geojson
import folium

india = folium.Map(location=[20, 77],tiles='Mapbox Bright', zoom_start=7)
india.save("india.html")

import shapefile
   # read the shapefile
reader = shapefile.Reader("IND_adm1.shx")
fields = reader.fields[1:]
field_names = [field[0] for field in fields]
buffer = []
for sr in reader.shapeRecords():
    atr = dict(zip(field_names, sr.record))
    geom = sr.shape.__geo_interface__
    buffer.append(dict(type="Feature", \
    geometry=geom, properties=atr)) 
   
   # write the GeoJSON file
from json import dumps
geojson = open("india.json", "w")
geojson.write(dumps({"type": "FeatureCollection","features": buffer}, indent=2) + "\n")
geojson.close()
