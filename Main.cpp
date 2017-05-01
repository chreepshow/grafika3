//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

//// vertex shader in GLSL
//const char *vertexSource = R"(
//	uniform mat4 M, Minv, MVP;
//	layout(location = 0) in vec3 vtxPos;
//	layout(location = 1) in vec3 vtxNorm;
//	out vec4 color;
//	
//	void main() {
//		gl_Position = vec4(vtxPos, 1) * MVP;
//		vec4 wPos = vec4(vtxPos, 1) * M;
//		vec4 wNormal = Minv * vec4(vtxNorm, 0);
//		color = Illumination(wPos, wNormal);
//	}
//)";
//
//
//// fragment shader in GLSL
//const char * fragmentSource = R"(
//	#version 330
//    precision highp float;
//
//	in vec3 color;				// variable input: interpolated color of vertex shader
//	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
//
//	void main() {
//		fragmentColor = vec4(color, 1); // extend RGB to RGBA
//	}
//)";

struct vec3 {
	float x, y, z;

	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

	vec3 operator+(const vec3& v) const {
		return vec3(x + v.x, y + v.y, z + v.z);
	}
	vec3 operator-(const vec3& v) const {
		return vec3(x - v.x, y - v.y, z - v.z);
	}
	vec3 operator*(const vec3& v) const {
		return vec3(x * v.x, y * v.y, z * v.z);
	}
	vec3 operator/(const vec3& v) const {
		return vec3(x / v.x, y / v.y, z / v.z);
	}
	vec3 operator-() const {
		return vec3(-x, -y, -z);
	}
	vec3 normalize() const {
		return (*this) * (1 / (Length() + 0.000001));
	}
	float Length() const { return sqrtf(x * x + y * y + z * z); }

	operator float*() { return &x; }
};

float dot(const vec3& v1, const vec3& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
	void SetUniform(unsigned shaderProg, char * name) {
		int loc = glGetUniformLocation(shaderProg, name);   	
		glUniformMatrix4fv(loc, 1, GL_TRUE, &m[0][0]);
	}
		
};

mat4 Translate(float tx, float ty, float tz) {
	return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			tx, ty, tz, 1);
}
		
mat4 Rotate(float angle, float wx, float wy, float wz) {
	vec3 w = vec3 (wx, wy, wz).normalize();
	vec3 i(1, 0, 0);
	vec3 j(0, 1, 0);
	vec3 k(0, 0, 1);
	vec3 resI = i*cosf(angle) + dot(w, (i - w))*(1 - cosf(angle)) + cross(w, i)*sinf(angle);
	vec3 resJ = j*cosf(angle) + dot(w, (i - w))*(1 - cosf(angle)) + cross(w, j)*sinf(angle);
	vec3 resK = k*cosf(angle) + dot(w, (i - w))*(1 - cosf(angle)) + cross(w, k)*sinf(angle);
	return mat4(resI.x, resI.y, resI.z, 0,
			resJ.x, resJ.y, resJ.z, 0,
			resK.x, resK.y, resK.z, 0,
			0, 0, 0, 1);
}
		
mat4 Scale(float sx, float sy, float sz) {
	return mat4(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, sz, 0,
			0, 0, 0, 1);
}

// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
};

struct Material {
	vec3 ka, kd, ks;
	float shine;
};

struct Texture {
	unsigned int textureId;
	Texture(char * fname) {
		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		int width, height;
		//float *image = LoadImage(fname, width, height); // megírni!
		float asdf[] = {1.0f,1.0f,1.0f};
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
			0, GL_RGB, GL_FLOAT, asdf); //Texture -> OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

struct Light {
	vec3 La, Le;
	vec3 wLightPos;
};

struct RenderState {
	mat4 M, V, P, Minv;
	Material * material;
	Texture *  texture;
	Light light;
	vec3 wEye;
};

struct Shader {
	unsigned int shaderProgram;

	void Create(const char * vsSrc,
		const char * fsSrc, const char * fsOuputName) {
		unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
		if (!vs) {
			printf("Error in vertex shader creation\n");
			exit(1);
		}
		glShaderSource(vs, 1, &vsSrc, NULL); glCompileShader(vs);
		checkShader(vs, "Vertex shader error");
		unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, &fsSrc, NULL); glCompileShader(fs);
		shaderProgram = glCreateProgram();
		glAttachShader(shaderProgram, vs);
		glAttachShader(shaderProgram, fs);

		glBindFragDataLocation(shaderProgram, 0, fsOuputName);
		glLinkProgram(shaderProgram);
	}
	virtual
		void Bind(RenderState& state) { glUseProgram(shaderProgram); }
};

class ShadowShader : public Shader {
	const char * vsSrc = R"(
		uniform mat4 MVP;
		layout(location = 0) in vec3 vtxPos;
		void main() { gl_Position = vec4(vtxPos, 1) * MVP; }
)";

	const char * fsSrc = R"(
		out vec4 fragmentColor;
		void main() { fragmentColor = vec4(1, 1, 1, 1); }
)";
public:
	ShadowShader() {
		
	}

	void Bind(RenderState& state) {
		glUseProgram(shaderProgram);
		mat4 MVP = state.M * state.V * state.P;
		MVP.SetUniform(shaderProgram, "MVP");
	}
	void Init() {
		Create(vsSrc, fsSrc, "fragmentColor");
	}
};

class PhongShader : public Shader {
	const char * vsSrc = R"(
	uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
	uniform vec4  wLiPos;       // pos of light source 
	uniform vec3  wEye;         // pos of eye

	layout(location = 0) in vec3 vtxPos; // pos in model sp
	layout(location = 1) in vec3 vtxNorm;// normal in mod sp

	out vec3 wNormal;           // normal in world space
	out vec3 wView;             // view in world space
	out vec3 wLight;            // light dir in world space

	void main() {
		gl_Position = vec4(vtxPos, 1) * MVP; // to NDC

		vec4 wPos = vec4(vtxPos, 1) * M;
		wLight  = wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w;
		wView   = wEye * wPos.w - wPos.xyz;
		wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
	}
)";

	const char * fsSrc = R"(
	uniform vec3 kd, ks, ka;// diffuse, specular, ambient ref
	uniform vec3 La, Le;    // ambient and point source rad
	uniform float shine;    // shininess for specular ref

	in  vec3 wNormal;       // interpolated world sp normal
	in  vec3 wView;         // interpolated world sp view
	in  vec3 wLight;        // interpolated world sp illum dir
	out vec4 fragmentColor; // output goes to frame buffer

	void main() {
		vec3 N = normalize(wNormal);
		vec3 V = normalize(wView);  
		vec3 L = normalize(wLight);
		vec3 H = normalize(L + V);
		float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
		vec3 color = ka * La + 
			       (kd * cost + ks * pow(cosd,shine)) * Le;
		fragmentColor = vec4(color, 1);
	}
)";
public:
	PhongShader() {

	}

	void Bind(RenderState& state) {
		glUseProgram(shaderProgram);
		mat4 MVP = state.M * state.V * state.P;
		MVP.SetUniform(shaderProgram, "MVP");
	}
	void Init() {
		Create(vsSrc, fsSrc, "fragmentColor");
	}
};

struct Geometry {
	unsigned int vao, nVtx;

	Geometry() {
		
	}
	void Draw() {
		glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, nVtx);
	}
	void Init() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
};

struct VertexData {
	vec3 position, normal;
	float u, v;
};

struct ParamSurface : Geometry {
	virtual VertexData GenVertexData(float u, float v) = 0;
	void Create(int N, int M);
};

void ParamSurface::Create(int N, int M) {
	nVtx = N * M * 6;
	unsigned int vbo;
	glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);

	VertexData *vtxData = new VertexData[nVtx], *pVtx = vtxData; //itt volt valami baja a vertexdata létrehozásával, talán az, h nem volt default konstruktora a vec3-nak
	for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) {
		*pVtx++ = GenVertexData((float)i / N, (float)j / M);
		*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
		*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
		*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
		*pVtx++ = GenVertexData((float)(i + 1) / N, (float)(j + 1) / M);
		*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
	}
	glBufferData(GL_ARRAY_BUFFER,
		nVtx * sizeof(VertexData), vtxData, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);  // AttribArray 0 = POSITION
	glEnableVertexAttribArray(1);  // AttribArray 1 = NORMAL
	glEnableVertexAttribArray(2);  // AttribArray 2 = UV
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
		sizeof(VertexData), (void*)offsetof(VertexData, position));
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
		sizeof(VertexData), (void*)offsetof(VertexData, normal));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
		sizeof(VertexData), (void*)offsetof(VertexData, u));
}

class Sphere : public ParamSurface {
	vec3 center;
	float radius;
public:
	Sphere(vec3 c, float r) : center(c), radius(r) {
		
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(cos(u * 2 * M_PI) * sin(v*M_PI),
			sin(u * 2 * M_PI) * sin(v*M_PI),
			cos(v*M_PI));
		vd.position = vd.normal * radius + center;
		vd.u = u; vd.v = v;
		return vd;
	}

	void SphereInit() {
		Create(16, 8); // tessellation level
	}
};

struct Camera {
	vec3  wEye, wLookat, wVup;
	float fov, asp, fp, bp;

	mat4 V() { // view matrix
		vec3 w = (wEye - wLookat).normalize();
		vec3 u = cross(wVup, w).normalize();
		vec3 v = cross(w, u);
		return Translate(-wEye.x, -wEye.y, -wEye.z) *
			mat4(u.x, v.x, w.x, 0.0f,
				u.y, v.y, w.y, 0.0f,
				u.z, v.z, w.z, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f);
	}
	mat4 P() { // projection matrix
		float sy = 1 / tan(fov / 2);
		return mat4(sy / asp, 0.0f, 0.0f, 0.0f,
			0.0f, sy, 0.0f, 0.0f,
			0.0f, 0.0f, -(fp + bp) / (bp - fp), -1.0f,
			0.0f, 0.0f, -2 * fp*bp / (bp - fp), 0.0f);
	}
};

Sphere sphere(vec3(0, 0, 0), 5);
ShadowShader sshader;
Material diff;

class Object {
	Shader *   shader;
	Material * material;
	Texture *  texture;
	Geometry * geometry;
	vec3 scale, pos, rotAxis;
	float rotAngle;
public:
	Object() {
		init();
	}
	virtual void Draw(RenderState state) {
		state.M = Scale(scale.x, scale.y, scale.z) *
			Rotate(rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
			Translate(pos.x, pos.y, pos.z);
		state.Minv = Translate(-pos.x, -pos.y, -pos.z) *
			Rotate(-rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
			Scale(1 / scale.x, 1 / scale.y, 1 / scale.z);
		state.material = material; state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}
	virtual void Animate(float dt) {}
	void init() {
		geometry = &sphere;
		shader = &sshader;
		diff.ka = vec3(0,1,0);
		diff.ks = vec3(1,1,1);
		diff.kd = vec3(1,0,0);
		diff.shine = 20;
		material = &diff;
		scale = vec3(2,2,2);
		pos = vec3(1, 1, 1);
		rotAxis = vec3(1, 1, 1);
		rotAngle = M_PI / 2;
	}
};

Object hardCodedSphere;
Camera initCamera() {
	Camera camera;
	camera.wEye = vec3(0, 0, 40);
	camera.wLookat = vec3(0, 0, 5);
	camera.wVup = vec3(0, 10, 5);
	camera.fov = 3.14 / 6;
	camera.asp = 1.0f;
	camera.fp = 2;
	camera.bp = 40;
	return camera;
}

class Scene {
	Camera camera;
	std::vector<Object *> objects;
	Light light;
	RenderState state;
public:
	Scene() {
		init();
	}
	void Render() {
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.light = light;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(float dt) {
		for (Object * obj : objects) obj->Animate(dt);
	}
	void init() {
		this->camera = initCamera();
		objects.push_back(&hardCodedSphere);
		light.La = vec3(0.1,0.1,0.1);
		light.Le = vec3(1,1,1);
		light.wLightPos = vec3(2, 2, 2);
	}
};


// handle of the shader program
//unsigned int shaderProgram;
Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	//glEnable(GL_DEPTH_TEST); // z-buffer is on
	//glDisable(GL_CULL_FACE); // backface culling is off

	sphere.Init();
	sphere.SphereInit();
	sshader.Init();
	//Create objects by setting up their vertex data on the GPU
	//Create vertex shader from string
	//unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	//if (!vertexShader) {
	//	printf("Error in vertex shader creation\n");
	//	exit(1);
	//}
	//glShaderSource(vertexShader, 1, &vertexSource, NULL);
	//glCompileShader(vertexShader);
	//checkShader(vertexShader, "Vertex shader error");

	//// Create fragment shader from string
	//unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	//if (!fragmentShader) {
	//	printf("Error in fragment shader creation\n");
	//	exit(1);
	//}
	//glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	//glCompileShader(fragmentShader);
	//checkShader(fragmentShader, "Fragment shader error");

	//// Attach shaders to a single program
	//shaderProgram = glCreateProgram();
	//if (!shaderProgram) {
	//	printf("Error in shader program creation\n");
	//	exit(1);
	//}
	//glAttachShader(shaderProgram, vertexShader);
	//glAttachShader(shaderProgram, fragmentShader);

	//// Connect the fragmentColor to the frame buffer memory
	//glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	//															// program packaging
	//glLinkProgram(shaderProgram);
	//checkLinking(shaderProgram);
	//// make this program run
	//glUseProgram(shaderProgram);
}

void onExit() {
	//glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	//triangle.Draw();
	//lineStrip.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		//lineStrip.AddPoint(cX, cY);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	//camera.Animate(sec);					// animate the camera
	//triangle.Animate(sec);					// animate the triangle object
	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
