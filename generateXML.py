import os
import xml.etree.ElementTree as et
from xml.dom import minidom
import numpy as np
import argparse
import glob
import scipy.io as io
import cv2

class tex():
    def __init__(self, diffuseName = None, roughnessName = None):
        self.diffuseName = diffuseName
        self.roughnessName = roughnessName

class mat():
    def __init__(self, name = 'mat', diffuse = None, roughness = None, texture = None):
        self.name = name
        self.diffuse = diffuse
        self.roughness = roughness
        self.texture = texture

############################# Code for Generating the Xml file ########################
def addShape(root, name, materials, isAddSpecular = True,
        isAddTransform = False,
        meshTranslate = None, meshRotateAxis = None, meshRotateAngle = None, meshScale = None,
        rotateAxis = None, rotateAngle = None,
        scaleValue = None, translationValue = None):
    shape = et.SubElement(root, 'shape' )
    shape.set('id', '{0}_object'.format(name.split('.')[0] ) )

    objType = name.split('.')[-1]
    assert(objType == 'ply' or objType == 'obj')
    shape.set('type', objType )
    stringF = et.SubElement(shape, 'string' )
    stringF.set('name', 'filename' )
    stringF.set('value', name )

    for material in materials:
        bsdf = et.SubElement(shape, 'bsdf' )

        if isAddSpecular == False:
            bsdf.set('type', 'diffuse' )
            if material.texture is None:
                rgb = et.SubElement(bsdf, 'rgb' )
                rgb.set('name', 'reflectance' )
                rgb.set('value', '%.5f %.5f %.5f'
                        % (material.diffuse[0], material.diffuse[1], material.diffuse[2] ) )
            else:
                diffPath = material.texture.diffuseName
                texture = et.SubElement(bsdf, 'texture' )
                texture.set('name', 'reflectance' )
                texture.set('type', 'bitmap' )
                filename = et.SubElement(texture, 'string' )
                filename.set('name', 'filename' )
                filename.set('value', diffPath )

        elif isAddSpecular == True:
            bsdf.set('type', 'microfacet')
            if material.texture is None:
                rgb = et.SubElement(bsdf, 'rgb')
                rgb.set('name', 'albedo')
                rgb.set('value', '%.5f %.5f %.5f'
                        % (material.diffuse[0], material.diffuse[1], material.diffuse[2]) )
                rgb = et.SubElement(bsdf, 'float')
                rgb.set('name', 'roughness')
                rgb.set('value', '%.5f' % (material.roughness ) )
            else:
                diffPath = material.texture.diffuseName
                texture = et.SubElement(bsdf, 'texture')
                texture.set('name', 'albedo')
                texture.set('type', 'bitmap')
                filename = et.SubElement(texture, 'string')
                filename.set('name', 'filename')
                filename.set('value', diffPath)

                roughPath = material.texture.roughnessName
                texture = et.SubElement(bsdf, 'texture')
                texture.set('name', 'roughness')
                texture.set('type', 'bitmap')
                filename = et.SubElement(texture, 'string')
                filename.set('name', 'filename')
                filename.set('value', roughPath)

    if isAddTransform:
        transform = et.SubElement(shape, 'transform')
        transform.set('name', 'toWorld')
        if not meshTranslate is None:
            translation = et.SubElement(transform, 'translate')
            translation.set('x', '%.5f' % meshTranslate[0] )
            translation.set('y', '%.5f' % meshTranslate[1] )
            translation.set('z', '%.5f' % meshTranslate[2] )
        if not meshRotateAxis is None:
            assert(not meshRotateAngle is None)
            rotation = et.SubElement(transform, 'rotate')
            rotation.set('x', '%.5f' % meshRotateAxis[0] )
            rotation.set('y', '%.5f' % meshRotateAxis[1] )
            rotation.set('z', '%.5f' % meshRotateAxis[2] )
            rotation.set('angle', '%.5f' % meshRotateAngle )
        if not meshScale is None:
            scale = et.SubElement(transform, 'scale')
            scale.set('value', '%.5f' % meshScale )
        if not rotateAxis is None:
            assert(not rotateAngle is None)
            rotation = et.SubElement(transform, 'rotate')
            rotation.set('x', '%.5f' % rotateAxis[0] )
            rotation.set('y', '%.5f' % rotateAxis[1] )
            rotation.set('z', '%.5f' % rotateAxis[2] )
            rotation.set('angle', '%.5f' % rotateAngle )
        if not scaleValue is None:
            scale = et.SubElement(transform, 'scale')
            scale.set('value', '%.5f' % scaleValue )
        if not translationValue is None:
            translation = et.SubElement(transform, 'translate')
            translation.set('x', '%.5f' % translationValue[0] )
            translation.set('y', '%.5f' % translationValue[1] )
            translation.set('z', '%.5f' % translationValue[2] )
    return root

def addEnv(root, envmapName, scaleFloat):
    emitter = et.SubElement(root, 'emitter')
    emitter.set('type', 'envmap')
    filename = et.SubElement(emitter, 'string')
    filename.set('name', 'filename')
    filename.set('value', envmapName )
    scale = et.SubElement(emitter, 'float' )
    scale.set('name', 'scale')
    scale.set('value', '%.4f' % (scaleFloat) )
    return root

def addSensor(root, fovValue, imWidth, imHeight, sampleCount):
    camera = et.SubElement(root, 'sensor')
    camera.set('type', 'perspective')
    fov = et.SubElement(camera, 'float')
    fov.set('name', 'fov')
    fov.set('value', '%.4f' % (fovValue) )
    fovAxis = et.SubElement(camera, 'string')
    fovAxis.set('name', 'fovAxis')
    fovAxis.set('value', 'x')
    transform = et.SubElement(camera, 'transform')
    transform.set('name', 'toWorld')
    lookAt = et.SubElement(transform, 'lookAt')
    lookAt.set('origin', '0 0 0')
    lookAt.set('target', '0 0 1.0')
    lookAt.set('up', '0 1.0 0')
    film = et.SubElement(camera, 'film')
    film.set('type', 'hdrfilm')
    width = et.SubElement(film, 'integer')
    width.set('name', 'width')
    width.set('value', '%d' % (imWidth) )
    height = et.SubElement(film, 'integer')
    height.set('name', 'height')
    height.set('value', '%d' % (imHeight) )
    sampler = et.SubElement(camera, 'sampler')
    sampler.set('type', 'adaptive')
    sampleNum = et.SubElement(sampler, 'integer')
    sampleNum.set('name', 'sampleCount')
    sampleNum.set('value', '%d' % (sampleCount) )
    return root

def transformToXml(root ):
    rstring = et.tostring(root, 'utf-8')
    pstring = minidom.parseString(rstring)
    xmlString = pstring.toprettyxml(indent="    ")
    xmlString= xmlString.split('\n')
    xmlString = [x for x in xmlString if len(x.strip()) != 0 ]
    xmlString = '\n'.join(xmlString )
    return xmlString

def generateXML(shapes, materials, envmapName, xmlName, sampleCount = 1024,
        imWidth = 640, imHeight = 480, fovValue = 63.4149,
        meshTranslate = None, meshRotateAxis = None, meshRotateAngle = None,
        meshScale = None, rotateAxis = None, rotateAngle = None,
        translation = None, scale = None ):

    # Build the scene
    root = et.Element('scene')
    root.set('version', '0.5.0')
    integrator = et.SubElement(root, 'integrator')
    integrator.set('type', 'path')

    rootObj = et.Element('scene')
    rootObj.set('version', '0.5.0')
    integrator = et.SubElement(rootObj, 'integrator')
    integrator.set('type', 'path')

    rootBkg = et.Element('scene')
    rootBkg.set('version', '0.5.0')
    integrator = et.SubElement(rootBkg, 'integrator')
    integrator.set('type', 'path')

    ## Create the obj files that is not emitter
    # Write 3D meshes
    root = addShape(root, shapes[0], [materials[0] ], True,
            isAddTransform = False)
    rootBkg= addShape(rootBkg, shapes[0], [materials[0] ], True,
            isAddTransform = False)

    root = addShape(root, shapes[1], [materials[1] ], True,
            isAddTransform = True,
            meshTranslate = meshTranslate, meshRotateAxis = meshRotateAxis,
            meshRotateAngle = meshRotateAngle, meshScale = meshScale,
            rotateAxis = rotateAxis, rotateAngle = rotateAngle,
            scaleValue = scale, translationValue = translation )
    rootObj = addShape(rootObj, shapes[1], [materials[1] ], True,
            isAddTransform = True,
            meshTranslate = meshTranslate, meshRotateAxis = meshRotateAxis,
            meshRotateAngle = meshRotateAngle, meshScale = meshScale,
            rotateAxis = rotateAxis, rotateAngle = rotateAngle,
            scaleValue = scale, translationValue = translation )

    # Add the environmental map lighting
    root = addEnv(root, envmapName, 1 )
    rootObj = addEnv(rootObj, envmapName, 1 )
    rootBkg = addEnv(rootBkg, envmapName, 1 )

    # Add the camera
    root = addSensor(root, fovValue, imWidth, imHeight, sampleCount)
    rootObj = addSensor(rootObj, fovValue, imWidth, imHeight, sampleCount)
    rootBkg = addSensor(rootBkg, fovValue, imWidth, imHeight, sampleCount)

    xmlString = transformToXml(root )
    xmlStringObj = transformToXml(rootObj )
    xmlStringBkg = transformToXml(rootBkg )

    with open(xmlName, 'w') as xmlOut:
        xmlOut.write(xmlString )
    with open(xmlName.replace('.xml', '_obj.xml'), 'w') as xmlOut:
        xmlOut.write(xmlStringObj )
    with open(xmlName.replace('.xml', '_bkg.xml'), 'w') as xmlOut:
        xmlOut.write(xmlStringBkg )


########################## Code for Rotating the Envmap ###########################
def angleToUV(theta, phi):
    u = (phi + np.pi) / 2 / np.pi
    v = 1 - theta / np.pi
    return u, v

def uvToEnvmap(envmap, u, v):
    height, width = envmap.shape[0], envmap.shape[1]
    c, r = u * (width-1), (1-v) * (height-1)
    cs, rs = int(c), int(r)
    ce = min(width-1, cs + 1)
    re = min(height-1, rs + 1)
    wc, wr = c - cs, r - rs
    color1 = (1-wc) * envmap[rs, cs, :] + wc * envmap[rs, ce, :]
    color2 = (1-wc) * envmap[re, cs, :] + wc * envmap[re, ce, :]
    color = (1 - wr) * color1 + wr * color2
    return color

def rotateEnvmap(envmap, vn ):
    up = np.array([0, 1, 0], dtype=np.float32 )
    z = vn
    z = z / np.sqrt(np.sum(z * z) )
    x = np.cross(up, z)
    x = x / np.sqrt(np.sum(x * x ) )
    y = np.cross(z, x)
    y = y / np.sqrt(np.sum(y * y) )

    #x = np.asarray([x[2], x[0], x[1]], dtype = np.float32 )
    #y = np.asarray([y[2], y[0], y[1]], dtype = np.float32 )
    #z = np.asarray([z[2], z[0], z[1]], dtype = np.float32 )
    x, y, z = x[np.newaxis, :], y[np.newaxis, :], z[np.newaxis, :]

    R = np.concatenate([x, y, z], axis=0)
    rx, ry, rz = R[:, 0], R[:, 1], R[:, 2]
    print(R)

    envmapRot = np.zeros(envmap.shape, dtype=np.float32)
    height, width = envmapRot.shape[0], envmapRot.shape[1]
    for r in range(0, height):
        for c in range(0, width):
            theta = r / float(height-1) * np.pi
            phi = (c / float(width) * np.pi * 2 - np.pi )
            z = np.sin(theta) * np.cos(phi)
            x = np.sin(theta) * np.sin(phi)
            y = np.cos(theta)
            coord = x * rx + y * ry + z * rz
            nx, ny, nz = coord[0], coord[1], coord[2]
            thetaNew = np.arccos(nz)
            nx = nx / (np.sqrt(1-nz*nz) + 1e-12)
            ny = ny / (np.sqrt(1-nz*nz) + 1e-12)
            nx = np.clip(nx, -1, 1)
            ny = np.clip(ny, -1, 1)
            nz = np.clip(nz, -1, 1)
            phiNew = np.arccos(nx)
            if ny < 0:
                phiNew = - phiNew
            u, v = angleToUV(thetaNew, phiNew)
            color = uvToEnvmap(envmap, u, v)
            envmapRot[r, c, :] = color

    return envmapRot



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampleCount', type=int, default=1024, help='the number of samples' )
    parser.add_argument('--fov', type=float, default=63.4149, help='the field of view in x axis')
    parser.add_argument('--envName', required=True, help='path to environmental map')
    parser.add_argument('--roughnessName', required = True, help='path to roughness')
    parser.add_argument('--diffuseName', required = True, help='path to roughness')
    parser.add_argument('--meshName', required=True, help='path to plane mesh')
    parser.add_argument('--meshNewName', required=True, help='path to the new object mesh')
    parser.add_argument('--infoName', required=True, help='the starting point')
    parser.add_argument('--rColor', default=0.8, type=float )
    parser.add_argument('--gColor', default=0.8, type=float )
    parser.add_argument('--bColor', default=0.8, type=float )
    parser.add_argument('--roughness', default=1.0, type=float )
    parser.add_argument('--meshTranslate', nargs=3, default=[0.0, 0.0, 0.0], type=float )
    parser.add_argument('--meshRotateAxis', nargs=3, default=[0.0, 1.0, 0.0], type=float )
    parser.add_argument('--meshRotateAngle', default=0.0, type=float )
    parser.add_argument('--meshScale', default=1.0, type=float )
    opt = parser.parse_args()
    print(opt)

    # Load information
    info = io.loadmat(opt.infoName )
    vn, vobj, vimg, scale = info['vn'], info['vobj'], info['vimg'], info['scale']
    vn, vobj, vimg = vn.flatten(), vobj.flatten(), vimg.flatten()

    vn = vn / np.sqrt(np.sum(vn * vn) )

    # Load environmental map
    env = np.load(opt.envName )['env']
    envRow, envCol = env.shape[0], env.shape[1]
    cId, rId = (envCol -1) * vimg[0], (envRow-1) * vimg[1]
    rId = np.clip(np.round(rId ), 0, envRow - 1)
    cId = np.clip(np.round(cId ), 0, envCol - 1)
    rId, cId = int(rId), int(cId )

    env = env[rId, cId, :, :, :]
    env = cv2.resize(env, (1024, 256), interpolation = cv2.INTER_LINEAR )
    envMatName = opt.envName.replace('.npz', 'Origin.mat')
    envDict = {}
    envDict['env'] = np.maximum(env, 0)
    io.savemat(envMatName, envDict )
    envBalck = np.zeros([256, 1024, 3], dtype=np.float32 )
    env = np.concatenate([env, envBalck], axis=0 )

    env = rotateEnvmap(env, vn)

    envDict = {}
    envDict['env'] = np.maximum(env, 0)
    envMatName = opt.envName.replace('.npz', '.mat' )
    envImName = envMatName.replace('mat', 'hdr')
    io.savemat(envMatName, envDict )

    # Load the roughness to check imsize
    im = cv2.imread(opt.roughnessName )
    imHeight, imWidth = im.shape[0], im.shape[1]

    # Build the materials for the two shapes
    shapes = [opt.meshName, opt.meshNewName ]
    mat1 = mat( texture = tex(diffuseName = opt.diffuseName, roughnessName = opt.roughnessName) )
    mat2 = mat(diffuse=[opt.rColor, opt.gColor, opt.bColor], roughness = opt.roughness)
    materials = [mat1, mat2 ]

    meshRotateAxis = np.array(opt.meshRotateAxis, dtype = np.float32 )
    meshRotateAxis = meshRotateAxis / np.sqrt(np.sum(meshRotateAxis * meshRotateAxis ) )

    xmlName = 'scene.xml'
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32 )
    rotateAxis = np.cross(up, vn)
    if np.sum(rotateAxis * rotateAxis) <= 1e-6:
        rotateAxis = None
        rotateAngle = None
    else:
        rotateAxis = rotateAxis / np.sqrt(np.sum(rotateAxis * rotateAxis) )
        rotateAngle = np.arccos(np.sum(vn * up ) )
    translation = vobj
    scale = scale

    generateXML(
        shapes = shapes, materials = materials,
        envmapName = envImName, xmlName = xmlName,
        sampleCount = opt.sampleCount,
        imWidth = imWidth, imHeight = imHeight, fovValue = opt.fov,
        meshTranslate = opt.meshTranslate, meshRotateAxis = meshRotateAxis,
        meshRotateAngle = opt.meshRotateAngle, meshScale = opt.meshScale,
        rotateAxis = rotateAxis, rotateAngle = rotateAngle,
        translation = translation, scale = scale )



