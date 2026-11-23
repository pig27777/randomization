"""Shared CPU studio with a depth-tested two-sided PE overlay."""
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pe-yam-film-comparison-r1"))
from render import make_model, raw_cmd, mj, W, H, crease_normals
from experience.yam_platform.render_video import GL
from OpenGL.GL import shaders

VERTEX = """
#version 120
varying vec3 normal;
varying vec3 position;
void main() {
    position = gl_Vertex.xyz;
    normal = gl_Normal;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
"""
FRAGMENT = """
#version 120
uniform vec3 eye;
uniform vec3 pigment;
uniform bool film;
varying vec3 normal;
varying vec3 position;
void main() {
    vec3 n = normalize(normal);
    vec3 view = normalize(eye - position);
    vec3 key = normalize(vec3(.44, -.61, .63));
    vec3 lamp = normalize(vec3(-.54, .38, .75));
    if (!film) {
        float diffuse = .47 + .43 * max(0., dot(n,key)) + .17 * max(0., dot(n,lamp));
        gl_FragColor = vec4(pigment * diffuse, 1.);
        return;
    }
    if (dot(n,view) < 0.) n = -n;
    float fresnel = pow(1. - clamp(dot(n,view),0.,1.), 3.3);
    float roughness = .12;
    vec3 reflected = reflect(-view,n);
    float sparkle = pow(max(dot(reflected,key),0.), 2. / (roughness*roughness))
                  + .56 * pow(max(dot(reflected,lamp),0.), 48.);
    vec3 tint = vec3(.93,.95,.98) * (.81 + .17 * max(dot(n,key),0.));
    tint += vec3(.52,.63,.77) * sparkle;
    gl_FragColor = vec4(tint, mix(.34,.80,fresnel));
}
"""

BOX_POINTS = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
BOX_FACES = np.array([
    [0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
    [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3],
])


def opaque_mesh(vertices, faces):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    # Native convex hull face orientation need not be consistent across assets.
    directions = np.sign(np.sum(normals * (triangles.mean(axis=1) - vertices.mean(axis=0)), axis=1))
    normals *= directions[:, None]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return triangles.reshape(-1, 3), np.repeat(normals, 3, axis=0)


def draw_mesh(vertices, normals):
    positions = np.ascontiguousarray(vertices, dtype=np.float32)
    directions = np.ascontiguousarray(normals, dtype=np.float32)
    GL.glVertexPointer(3, GL.GL_FLOAT, 0, positions)
    GL.glNormalPointer(GL.GL_FLOAT, 0, directions)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(positions))


def layer(renderer, model, data, view, option, program, mesh, contents, pigment):
    renderer.update_scene(data, camera=view, scene_option=option)
    scene = renderer.scene
    scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = False
    mj.mjr_render(renderer._rect, scene, renderer._mjr_context)
    camera = mj.mjv_averageCamera(scene.camera[0], scene.camera[1])
    # Keep the camera matrices and zero-to-one reversed depth of the CAD pass.
    GL.glUseProgram(program)
    GL.glDisable(GL.GL_LIGHTING)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDepthFunc(GL.GL_GEQUAL)
    GL.glDepthMask(GL.GL_TRUE)
    GL.glDisable(GL.GL_CULL_FACE)
    GL.glDisable(GL.GL_BLEND)
    GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
    GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
    GL.glUniform3fv(GL.glGetUniformLocation(program, "eye"), 1, camera.pos)
    GL.glUniform3fv(GL.glGetUniformLocation(program, "pigment"), 1, pigment)
    material = GL.glGetUniformLocation(program, "film")
    GL.glUniform1i(material, 0)
    for vertices, normals in contents:
        draw_mesh(vertices, normals)
    GL.glUniform1i(material, 1)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    GL.glDepthMask(GL.GL_FALSE)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glFrontFace(GL.GL_CCW)
    for cull in (GL.GL_FRONT, GL.GL_BACK):
        GL.glCullFace(cull)
        draw_mesh(*mesh)
    GL.glDisable(GL.GL_CULL_FACE)
    GL.glDisable(GL.GL_BLEND)
    GL.glDepthMask(GL.GL_TRUE)
    GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
    GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
    GL.glUseProgram(0)
    rgb = np.empty((H, W, 3), dtype=np.uint8)
    mj.mjr_readPixels(rgb, None, renderer._rect, renderer._mjr_context)
    return Image.fromarray(rgb[::-1])


def render(output, films, faces, links, quats, names, pads, contents, content_faces, pigment, *, still=None, grasp_detail=False):
    model = make_model(films[0], faces)
    model.geom_group[model.geom("surface").id] = 5
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    option = mj.MjvOption()
    option.geomgroup[[3, 5]] = 0
    option.sitegroup[:] = 0
    full = mj.MjvCamera()
    full.azimuth, full.elevation, full.distance = 135, -19, 1.34
    full.lookat[:] = [-.18, -.25, .237]
    detail = mj.MjvCamera()
    if grasp_detail:
        detail.azimuth, detail.elevation, detail.distance = 90, -26, .30
        inset_size, inset_origin = (396, 297), (18, 335)
    else:
        detail.azimuth, detail.elevation, detail.distance = 135, -26, .59
        inset_size, inset_origin = (300, 225), (18, 412)
    inset_x, inset_y = inset_origin
    inset_w, inset_h = inset_size
    bodies = {i: list(names).index(model.body(i).name) for i in range(model.nbody) if model.body(i).name in names}
    parent = [(g, bodies[b]) for g, b in enumerate(model.geom_bodyid) if b in bodies]
    geom_rotation = np.empty((model.ngeom, 9))
    for g, _ in parent:
        mj.mju_quat2Mat(geom_rotation[g], model.geom_quat[g])
    rotations = np.empty((len(names), 9))
    with mj.Renderer(model,H,W) as renderer:
        program = shaders.compileProgram(shaders.compileShader(VERTEX,GL.GL_VERTEX_SHADER),
                                         shaders.compileShader(FRAGMENT,GL.GL_FRAGMENT_SHADER))
        process = None if still is not None else subprocess.Popen(raw_cmd(output), stdin=subprocess.PIPE)
        try:
            for i in ([still] if still is not None else range(len(films)-1)):
                for b, orientation in enumerate(quats[i]):
                    mj.mju_quat2Mat(rotations[b],orientation)
                for g, b in parent:
                    r = rotations[b].reshape(3,3)
                    data.geom_xpos[g] = links[i,b] + r @ model.geom_pos[g]
                    data.geom_xmat[g] = (r @ geom_rotation[g].reshape(3,3)).ravel()
                v = films[i]
                membrane = (v[faces].reshape(-1, 3), crease_normals(v, faces))
                insides = [opaque_mesh(cv, cf) for cv, cf in zip(contents[i], content_faces)]
                frame = layer(renderer, model, data, full, option, program, membrane, insides, pigment)
                detail.lookat[:] = pads[i].mean(axis=0) + [0, 0, .002] if grasp_detail else v.mean(axis=0)
                inset = layer(renderer, model, data, detail, option, program, membrane, insides, pigment)
                frame.paste(inset.resize(inset_size, Image.Resampling.LANCZOS), inset_origin)
                ImageDraw.Draw(frame).rectangle((inset_x - 1, inset_y - 1, inset_x + inset_w, inset_y + inset_h),
                                                outline="#d6dce2")
                if process:
                    process.stdin.write(frame.tobytes())
                else:
                    frame.save(output)
        finally:
            GL.glDeleteProgram(program)
            if process:
                process.stdin.close()
                if process.wait():
                    raise RuntimeError(f"Encoding failed for {output}")
