import cv2
import numpy as np
import sys
import nvector as nv


#define the camera intrinsics, camera 35 mm focal length 20...NOT NECESSARY HERE
K_matrix = np.array([[1, 0, 0], [0, 1, 0], [0,0,1]])
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

#generate SIFT feauture extractor
sift = cv2.xfeatures2d.SIFT_create(500, 3, 0.04, 10, 1.6)



#testing function to draw feature matches between corresponding images
def draw_matches(image, points1, points2):
    #darray=[]
    for (x1,y1),(x2,y2) in zip(np.int32(points1), np.int32(points2)):
        cv2.line(image, (x1,y1), (x2,y2), (255, 0, 255), lineType = cv2.LINE_AA)

    return image

#extract and match features between each pair
def extract_and_match(image1, image2):

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    pt_image1, des_image1 = sift.detectAndCompute(image1, mask=None)
    pt_image2, des_image2 = sift.detectAndCompute(image2, mask=None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_image1,des_image2,k=2)


    good = []
    image1_pts = []
    image2_pts = []

    for i,(m_n) in enumerate(matches):
        if len(m_n) != 2:
           continue

        (m,n) = m_n
        if m.distance < 0.6*n.distance:
            good.append(m)
            image2_pts.append(pt_image2[m.trainIdx].pt)
            image1_pts.append(pt_image1[m.queryIdx].pt)


            image2_pts_np = np.int32(image2_pts)
            image1_pts_np = np.int32(image1_pts)
    darray = []

    #remove wild outliers
    for (x1,y1),(x2,y2) in zip(np.int32(image1_pts), np.int32(image2_pts)):
         d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
         darray.append(d)

    print(darray)

    mean = np.mean(darray)
    std =  np.std(darray)
    print(mean,std)
    index_list = []
    for i in xrange(len(darray)):
        if abs(darray[i]) > (abs(mean) + 0.5*std):
            index  = i
            index_list.append(i)

    print(index_list)
    shifter = 0
    for i in xrange(len(index_list)):
        image1_pts.remove(image1_pts[index_list[i]-shifter])
        image2_pts.remove(image2_pts[index_list[i]-shifter])
        darray.remove(darray[index_list[i-shifter]])
        shifter = shifter+1

    print(darray)

    return image1_pts, image2_pts



#initiate data matrix....ideally this should be a text file
data = [['dji_0644.jpg',-123.114661,38.426805,90.689292,9.367337,1.260910,0.385252],

['dji_0645.jpg',-123.114650,38.426831,90.825989,85.055542,-0.336052,1.667057],

['dji_0646.jpg',-123.114429,38.426830,91.088004,88.858391,-0.070967,1.876991],

['dji_0647.jpg',-123.114125,38.426831,91.091265,88.269956,0.671020,1.849037],

['dji_0648.jpg',-123.114104,38.426832,90.747063,184.433167,-1.492852,1.134858],

['dji_0649.jpg',-123.114136,38.426609,91.304548,190.422786,-0.656365,1.312138],

['dji_0650.jpg',-123.114203,38.426195,91.007241,190.053859,0.363708,1.444969],

['dji_0651.jpg',-123.114271,38.425813,91.538639,190.037347,1.106723,1.521566],

['dji_0652.jpg',-123.114284,38.425752,90.900331,190.344637,1.424554,1.632872],

['dji_0653.jpg',-123.114268,38.425751,90.622088,89.052669,1.243665,-1.090830],

['dji_0654.jpg',-123.113839,38.425752,91.235595,88.392906,1.794960,-0.221090],

['dji_0655.jpg',-123.113745,38.425749,90.437221,87.186642,1.947206,0.394757],

['dji_0656.jpg',-123.113734,38.425779,90.163445,6.838638,0.624994,-0.674300],

['dji_0657.jpg',-123.113662,38.426160,91.160272,6.815734,0.945930,0.550999],

['dji_0658.jpg',-123.113591,38.426581,91.454023,8.740611,1.059218,1.088282],

['dji_0659.jpg',-123.113556,38.426807,91.221973,9.253228,1.353285,1.449262],

['dji_0660.jpg',-123.113544,38.426829,90.324952,146.612422,-1.948292,0.194904],

['dji_0661.jpg',-123.113439,38.426665,90.864808,155.415639,-0.917097,1.375369],

['dji_0662.jpg',-123.113183,38.426287,91.956351,155.074334,0.208305,2.160615],

['dji_0663.jpg',-123.113116,38.426189,90.561950,153.763228,0.793427,2.490934],

['dji_0664.jpg',-123.113115,38.426165,90.604094,187.491139,-0.312975,2.836182],

['dji_0665.jpg',-123.113176,38.425826,91.781148,188.845376,0.574889,3.010090],

['dji_0666.jpg',-123.113185,38.425756,91.069673,189.163989,0.764728,2.785707],

['dji_0667.jpg',-123.113198,38.425754,90.750004,301.431548,-2.034127,0.511803]]


#rectify image orientation
yaw = np.array([[np.cos(np.deg2rad(data[0][4])), -np.sin(np.deg2rad(data[0][4])), 0], [np.sin(np.deg2rad(data[0][4])), np.cos(np.deg2rad(data[0][4])), 0], [0, 0, 1]])
pitch = np.array([[np.cos(np.deg2rad(data[0][5])), 0, np.sin(np.deg2rad(data[0][5]))],[0,1,0],[-np.sin(np.deg2rad(data[0][5])), 0, np.cos(np.deg2rad(data[0][5]))]])
roll = np.array([[1, 0, 0,], [0, np.cos(np.deg2rad(data[0][6])), -np.sin(np.deg2rad(data[0][6]))], [0, np.sin(np.deg2rad(data[0][6])), np.cos(np.deg2rad(data[0][6]))]])

#calculate all image rotations and compile composite R_matrix
R_Matrix = []
for i in xrange(len(data)):
    R_1_1 = np.cos(np.deg2rad(data[i][5]))*np.cos(np.deg2rad(data[i][4]))
    R_1_2 = np.cos(np.deg2rad(data[i][5]))*np.sin(np.deg2rad(data[i][4]))
    R_1_3 = -np.sin(np.deg2rad(data[i][5]))

    R_2_1 = np.cos(np.deg2rad(data[i][4]))*np.sin(np.deg2rad(data[i][6]))*np.sin(np.deg2rad(data[i][5]))-np.cos(np.deg2rad(data[i][6]))*np.sin(np.deg2rad(data[i][4]))
    R_2_2 = np.cos(np.deg2rad(data[i][6]))*np.cos(np.deg2rad(data[i][4]))+np.sin(np.deg2rad(data[i][6]))*np.sin(np.deg2rad(data[i][4]))*np.sin(np.deg2rad(data[i][5]))
    R_2_3 = np.cos(np.deg2rad(data[i][5]))*np.sin(np.deg2rad(data[i][6]))

    R_3_1 = np.sin(np.deg2rad(data[i][6]))*np.sin(np.deg2rad(data[i][4]))+np.cos(np.deg2rad(data[i][6]))*np.cos(np.deg2rad(data[i][4]))*np.sin(np.deg2rad(data[i][5]))
    R_3_2 = np.cos(np.deg2rad(data[i][6]))*np.sin(np.deg2rad(data[i][4]))*np.sin(np.deg2rad(data[i][5]))-np.cos(np.deg2rad(data[i][4]))*np.sin(np.deg2rad(data[i][6]))
    R_3_3 = np.cos(np.deg2rad(data[i][6]))*np.cos(np.deg2rad(data[i][5]))

    R = np.array([[R_1_1, R_1_2, R_1_3],[R_2_1, R_2_2, R_2_3],[R_3_1, R_3_2, R_3_3]])


    R_Matrix.append(R)


print(R_Matrix[0])

#NOTE: CHANGE PATH TO LOAD YOUR OWN DIRECTORY
path = '/Users/jacob/drone_deploy/example/'

#NOT USED: an attempt at aligning the images via the world coordinates given (normalized)
#normalize the 3-D coords to scale image coordinates
def normalize(val, minval, maxval):
    dst = np.abs(np.abs(maxval) - np.abs(minval))
    dst = maxval - minval
    return (np.abs(maxval) - np.abs(val)) / dst

wgs84 = nv.FrameE(name='WGS84')
x=[]
y=[]
z=[]

a = 6378.137000*(10**3)
WGS_E = 0.0818191908426
f=1/298.257223563
e = f*(2-f)

lon0, lat0 = np.deg2rad(-123.113198),np.deg2rad(38.425754)

def transformEcefToEnu(originLonLatAlt, ecef):
    """
    Transform tuple ECEF x,y,z (meters) to tuple E,N,U (meters).

    Based on http://en.wikipedia.org/wiki/Geodetic_system
    """
    lon, lat, alt = originLonLatAlt
    x, y, z = ecef
    pointA = wgs84.GeoPoint(lon, lat, alt, degrees=True)
    p_EA_E = pointA.to_ecef_vector()
    frame_N = nv.FrameN(pointA)
    p_EA_E = p_EA_E.pvector.ravel()
    ox = p_EA_E[0]
    oy = p_EA_E[1]
    oz = p_EA_E[2]
    dx, dy, dz = (x - ox, y - oy, z - oz)
    lonDeg, latDeg, _ = originLonLatAlt
    lon = np.deg2rad(lonDeg)
    lat = np.deg2rad(latDeg)
    return (-np.sin(lon) * dx + np.cos(lon) * dy,
            -np.sin(lat) * np.cos(lon) * dx - np.sin(lat) * np.sin(lon) * dy + np.cos(lat) * dz,
            np.cos(lat) * np.cos(lon) * dx + np.cos(lat) * np.sin(lon) * dy + np.sin(lat) * dz)

for i in xrange(len(data)):
    xindv = data[i][1]
    yindv = data[i][2]
    zindv = data[i][3]
    #x=lon, y=lat
    pointA = wgs84.GeoPoint(xindv,yindv,zindv, degrees=True)
    pointB = wgs84.GeoPoint(-123.1139,38.4263,90.750004, degrees=True)
    p_AB_E = nv.diff_positions(pointA, pointB)
    frame_N = nv.FrameN(pointA)


    p_AE_E = p_AB_E.change_frame(frame_N)
    p_EA_E = p_AE_E.pvector.ravel()
    x_EA = p_EA_E[0]
    y_EA = p_EA_E[1]
    z_EA = p_EA_E[2]
    x.append(x_EA)
    y.append(y_EA)
    z.append(z_EA)

# for i in xrange(len(x)):
#     x[i], y[i], z[i] = transformEcefToEnu((-123.1139,38.4263,90.750004), (x[i],y[i],z[i]))

print(y)
print(z)


focal=20.0
# M = []
Cart_to_Camera = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
for i in xrange(len(x)):
    x[i] = x[i]
    y[i] = y[i]
    R2D_image1 = np.float32([[R_Matrix[i][0][0],R_Matrix[i][0][1],0],[R_Matrix[i][1][0],R_Matrix[i][1][1], 0], [0,0,1]])
    Rotated = -np.linalg.inv(R2D_image1).dot([x[i],y[i],z[i]])
    x[i] = Rotated[0]
    y[i] = Rotated[1]
#     print (x[i],y[i])
print(x)
print(y)

min_x, min_y = min(x), min(y)
max_x, max_y = max(x), max(y)
#scale=850
for i in xrange(len(x)):
    x[i] = (x[i]*0.035)/z[i]
    y[i] = (y[i]*0.035)/z[i]
    x[i] = normalize(x[i], min_x, max_x)
    y[i] = normalize(y[i], min_y, max_y)
    #M[i] =

    # print(y[i])
print(x)
print(y)
#sys.exit()

#define the rotation 2D of teh image given the yaw, pitch, roll and the image center
def rotate_and_translate_image(path, data, R_Matrix, x, y):
    image1 = cv2.imread(path, cv2.IMREAD_COLOR)

    #resize image for processing
    r = 680.0 /image1.shape[1]
    dim = (680, int(image1.shape[0]*r))
    image1_resized = cv2.resize(image1,dim,interpolation=cv2.INTER_AREA)

    #create bounding box for rotation
    h,w = image1_resized.shape[:2]
    r = np.sqrt(w**2 + h**2)
    #print(r, w, h)
    blank = cv2.copyMakeBorder(image1_resized,int(np.ceil((r-h)/2)), int(np.ceil((r-h)/2)),int(np.ceil((r-w)/2)), int(np.ceil((r-w)/2)), cv2.BORDER_CONSTANT, 0)


    h,w = blank.shape[:2]

    #use width and height of image to translate rotation to center of new image
    R2D_image1 = np.float32([[R_Matrix[0][0],R_Matrix[0][1],h/2],[R_Matrix[1][0],R_Matrix[1][1], w/2], [0,0,1]])
    R2D_inv =  np.linalg.inv(R2D_image1)
    R_affine = np.float32([[R2D_inv[0][0], R2D_inv[0][1],R2D_inv[0][2]+w/2],[R2D_inv[1][0],R2D_inv[1][1],R2D_inv[1][2]+h/2]])
    dst1 = cv2.warpAffine(blank, R_affine, (h,w))

    return dst1

rotated_images = []
translated_images = []


#rotate images using given pitch, yaw, roll
for i in xrange(len(data)):
    path_to_file = path+data[i][0]
    image1 = rotate_and_translate_image(path_to_file, data[i], R_Matrix[i], x[i],y[i])
    rotated_images.append(image1)
    print(image1.shape[:2])


def translate_images(rotated_image, x, y):
    h,w = rotated_image.shape[:2]
    T = np.float32([[1,0,np.int(x)],[0,1,np.int(y)]])
    dst1_translated = cv2.warpAffine(rotated_image, T, (h,w))
    return dst1_translated




#test image of the rotation result
#image_two = cv2.addWeighted(rotated_images[0],0.5,rotated_images[1],0.5,0)

# cv2.imshow('image', image_two)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# sys.exit()

def homography_transform(rotated_images1, rotated_images2):
    dst1 = rotated_images1
    dst2 = rotated_images2
    h, w = dst1.shape[:2]

    #compute matching features from two image pairs
    image1_pts, image2_pts = extract_and_match(dst1, dst2)


    # #code to output the features matched in each image
    # image = cv2.addWeighted(rotated_images1,0.5,rotated_images2,0.5,0)
    # image = draw_matches(image, image1_pts, image2_pts)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()


    print(image1_pts)
    print(image2_pts)
    n = len(image1_pts)
    image2_pts_np = np.reshape(np.float32(image2_pts),(1,n,2))
    image1_pts_np = np.reshape(np.float32(image1_pts),(1,n,2))

    #compute homography transform from img1 to img2
    if(len(image1_pts)>=4):
        homography, mask = cv2.findHomography(np.float32(image1_pts), np.float32(image2_pts), cv2.RANSAC, 1.0)
        affine = cv2.estimateRigidTransform(np.float32(image1_pts), np.float32(image2_pts), fullAffine=True)
        print(homography)
        return homography, affine
    else:
        return None, None


#create the warped image overlay, given homography and a translation to avoid cropping
def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]])
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]])
    list_of_points_1 = np.array([list_of_points_1])
    temp_points = np.array([temp_points])
    #H =  np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
    #list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points_2 = cv2.transform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)


    [x_min, y_min] = np.min(np.min(list_of_points, axis=1), axis=0)
    [x_max, y_max] = np.max(np.max(list_of_points, axis=1), axis=0)

    print(list_of_points)
    print(x_min, y_min)

    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0,translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    #float d = H[0][0], H[0][1], H[0][2];

    #H_affine = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
    #H_translation_affine = np.array([[H_translation[0][0], H_translation[0][1], H_translation[0][2]], [H_translation[1][0], H_translation[1][1], H_translation[1][2]]])

    H_translation_affine = np.float32([[1,0,0],[0,1,0]])
    #H_affine = np.array([[H[0][0], H[0][1], H[0][2] + H_translation[0][2]], [H[1][0], H[1][1], H[1][2]+ H_translation[1][2]]])
    H_affine = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
    #output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))

    #img1_large = cv2.warpPerspective(img1, H_translation, (x_max-x_min, y_max-y_min))
    #output_img = cv2.warpPerspective(img2, H_translation.dot(H), (y_max-y_min, x_max-x_min))
    #img1_large = cv2.warpPerspective(img1, H_translation, (y_max-y_min, x_max-x_min))

    output_img = cv2.warpAffine(img2, H_affine, (y_max-y_min, x_max-x_min))
    img1_large = cv2.warpAffine(img1, H_translation_affine, (y_max-y_min, x_max-x_min))

    print(output_img.shape[:2])

    base_image = np.zeros((x_max-x_min, y_max-y_min, 3), np.uint8)

    print(base_image.shape[:2])

    (ret,data_map) = cv2.threshold(cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY),0, 255, cv2.THRESH_BINARY)

    base_image = cv2.add(base_image, img1_large, mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)

    final_img = cv2.add(base_image, output_img, dtype=cv2.CV_8U)

    return final_img

def crop_mosaic(final_img):
        #Crop off the black edges
    final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
    __,contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print "Found %d contours..." % (len(contours))

    max_area = 0
    best_rect = (0,0,0,0)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # print "Bounding Rectangle: ", (x,y,w,h)

        deltaHeight = h-y
        deltaWidth = w-x

        area = deltaHeight * deltaWidth

        if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
            max_area = area
            best_rect = (x,y,w,h)

        if ( max_area > 0 ):
            print "Maximum Contour: ", max_area
            print "Best Rectangle: ", best_rect

    final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],
                        best_rect[0]:best_rect[0]+best_rect[2]]

    return final_img_crop



#H image15-18 into image 14
___, homography_test1415 = homography_transform(rotated_images[15],rotated_images[14])
#image = cv2.warpAffine(homography_test, T, base_image.shape[:2])
image1 = warpImages(rotated_images[14], rotated_images[15], homography_test1415)
__, homography_test1416 = homography_transform(rotated_images[16],rotated_images[14])
image2 = warpImages(image1,rotated_images[16], homography_test1416)
__, homography_test1417 = homography_transform(rotated_images[17],rotated_images[14])
image1 = warpImages(image2,rotated_images[17], homography_test1417)
__, homography_test1418 = homography_transform(rotated_images[18],rotated_images[14])
image = warpImages(image1,rotated_images[18], homography_test1418)

#H1418 into H1819
__, homography_test1819 = homography_transform(rotated_images[19],rotated_images[18])
H1914 = np.vstack([homography_test1418, [0, 0, 1]]).dot(np.vstack([homography_test1819, [0, 0, 1]]))
print(H1914)
H1914 = np.array([[H1914[0][0],H1914[0][1],H1914[0][2]],[H1914[1][0],H1914[1][1],H1914[1][2]]])
image = warpImages(image,rotated_images[19], H1914)

#H1418 into H1820
__, homography_test1820 = homography_transform(rotated_images[20],rotated_images[18])
H2014 = np.vstack([homography_test1418, [0, 0, 1]]).dot(np.vstack([homography_test1820, [0, 0, 1]]))
H2014 = np.array([[H2014[0][0],H2014[0][1],H2014[0][2]],[H2014[1][0],H2014[1][1],H2014[1][2]]])
image = warpImages(image,rotated_images[20], H2014)

#H1418 into H1821
__, homography_test1821 = homography_transform(rotated_images[21],rotated_images[18])
H2114 = np.vstack([homography_test1418, [0, 0, 1]]).dot(np.vstack([homography_test1821, [0, 0, 1]]))
H2114 = np.array([[H2114[0][0],H2114[0][1],H2114[0][2]],[H2114[1][0],H2114[1][1],H2114[1][2]]])
image = warpImages(image,rotated_images[21], H2114)

#H1421 into H2122
__, homography_testH2122 = homography_transform(rotated_images[22],rotated_images[21])
H2214 = np.vstack([H2114, [0, 0, 1]]).dot(np.vstack([homography_testH2122, [0, 0, 1]]))
H2214 = np.array([[H2214[0][0],H2114[0][1],H2214[0][2]],[H2214[1][0],H2214[1][1],H2214[1][2]]])
image = warpImages(image,rotated_images[22], H2214)

#H1422 into H2223
__, homography_test2223 = homography_transform(rotated_images[23],rotated_images[22])
H2314 = np.vstack([H2214, [0, 0, 1]]).dot(np.vstack([homography_test2223, [0, 0, 1]]))
H2314 = np.array([[H2314[0][0],H2114[0][1],H2314[0][2]],[H2314[1][0],H2314[1][1],H2314[1][2]]])
image = warpImages(image,rotated_images[23], H2314)


image = crop_mosaic(image)
new_path = '/Users/jacob/drone_deploy/'
file_name = 'homography_test_1423.jpg'
file_name = new_path+file_name
cv2.imwrite(file_name, image)


file_name = 'homography_test_1417.jpg'
file_name = new_path+file_name
cv2.imwrite(file_name, image1)


