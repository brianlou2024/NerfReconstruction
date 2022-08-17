import os
import sqlite3
import numpy as np
import shutil

import configargparse

def sql():
    
    dbPath = f"{args.datadir}/database.db"
    params = np.genfromtxt(args.calib, delimiter=",")
    # params = [np.array((699.316, 665.998, 378.275, -0.171522, 0.0244116)),
    #             np.array((700.742, 628.812, 362.365, -0.173764, 0.0265925))]
    print(params)

    db = sqlite3.connect(dbPath)
    for i in range(len(params)):
        sqlCommand = "UPDATE cameras SET params=(?), prior_focal_length=1 WHERE camera_id=%d" % (i+1)
        db.execute(sqlCommand, (np.getbuffer(params[i]),))
    db.commit() # save data back to file
    db.close()

def edit_images(basedir):
    print("Deleting / Editing images directory...")
    model_path = os.path.join(basedir, "sparse/0/images.txt")
    images_path = os.path.join(basedir, "images")
    images = set()
    with open(model_path, "r") as f:
        lines = f.readlines()
        for i in range(4,len(lines)):
            if i%2 == 0:
                images.add(lines[i].split(" ")[-1].strip())
    count = 0
    for dir_, _, files in os.walk(images_path):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, images_path)
            rel_file = os.path.join(rel_dir, file_name)
            if str(rel_file) not in images:
                count += 1
                os.remove(os.path.join(dir_, file_name))
    print(f"Finished. Deleted {count} images")
    if count != 0:
        shutil.rmtree(os.path.join(basedir, "sparse"))
        os.remove(os.path.join(basedir, "database.db"))
    print("Re-running COLMAP Next (by deleting database.db)")




def main():
    ps = configargparse.ArgumentParser()
    ps.add_argument("--datadir", type=str, help='data directory')
    ps.add_argument("--calib", type=str, help='calibration file')
    ps.add_argument("--operation", type=str, choices=['calib','remove_images'],help='operation')
    args = ps.parse_args()

    if args.operation == "remove_images":
        edit_images(args.datadir)
    elif args.operation == "calib":
        sql()

if __name__ =='__main__':
    main()
