import util


db_path = '/home/amirreza/caffe/ekf/data/data_ntot_129600'
db = util.DartDB(db_path)

print db.length, ' instances are in the database.'

try:
    db.read_mean_img()
except Exception as e:
    print e
    

for i in range(db.length):
    try:
	db.read_instance(i)
	if i % 10000 == 0:
	    print i, 'images read'
    except Exception as e:
	print e