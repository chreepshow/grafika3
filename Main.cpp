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
// Nev    : Koppany Peter
// Neptun : Y9THTW
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

struct vec3 {
	float x, y, z;

	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

	vec3 operator/(float a) const { return vec3(x / a, y / a, z / a); }

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
	void SetUniform(unsigned int shaderProg, char* name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniform3f(loc, x, y, z);
	}
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
	vec3 w = vec3(wx, wy, wz).normalize();
	vec3 i(1, 0, 0);
	vec3 j(0, 1, 0);
	vec3 k(0, 0, 1);
	vec3 resI = i*cosf(angle) + (w*dot(i, w))*(1 - cosf(angle)) + cross(w, i)*sinf(angle);
	vec3 resJ = j*cosf(angle) + (w*dot(i, w))*(1 - cosf(angle)) + cross(w, j)*sinf(angle);
	vec3 resK = k*cosf(angle) + (w*dot(i, w))*(1 - cosf(angle)) + cross(w, k)*sinf(angle);
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

	vec4(vec3 vec, float w) {
		v[0] = vec.x; v[1] = vec.y; v[2] = vec.z; v[3] = w;
	}

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
	void SetUniform(unsigned int shaderProg, char * name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniform4f(loc, v[0], v[1], v[2], v[3]);
	}
};

struct CatmullRom {
	std::vector<vec3> cps;		// control points 
	std::vector<float>  ts;	// parameter (knot) values
	vec3 v0;
	vec3 vn;

	vec3 Hermite(vec3 p0, vec3 v0, float t0,
		vec3 p1, vec3 v1, float t1,
		float t) {
		vec3 res, a0, a1, a2, a3;
		a0 = p0;
		a1 = v0;
		a2 = (p1 - p0) * 3.0f / pow((t1 - t0), 2) - (v1 + v0 * 2.0f) / (t1 - t0);
		a3 = (p0 - p1) * 2.0f / pow((t1 - t0), 3) + (v1 + v0) / pow((t1 - t0), 2);
		float dt = t - t0;
		res = a3*pow(dt, 3) + a2*pow(dt, 2) + a1*dt + a0;
		return res;
	}

public:
	CatmullRom(){}
	CatmullRom(vec3 startV, vec3 endV) {
		v0 = startV;
		vn = endV;
	}
	void addControlPoint(vec3 cp, float t) {
		cps.push_back(cp);
		ts.push_back(t);
	}
	void erease() {
		cps.clear();
		ts.clear();
	}
	vec3 r(float t) {
		for (int i = 0; i < cps.size() - 1; i++) {
			if (ts[i] <= t && t <= ts[i + 1]) {
				vec3 vi, vii;
				if (i == 0 || i == cps.size() - 2) {
					if (i == 0) {
						vi = v0;
						vii = ((cps[i + 2] - cps[i + 1]) / (ts[i + 2] - ts[i + 1]) + (cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]))*0.5f;
					}
					else if (i == cps.size() - 2) {
						vi = ((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]) + (cps[i] - cps[i - 1]) / (ts[i] - ts[i - 1]))*0.5f;
						vii = vn;
					}
				}
				else {
					vi = ((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]) + (cps[i] - cps[i - 1]) / (ts[i] - ts[i - 1]))*0.5f;
					vii = ((cps[i + 2] - cps[i + 1]) / (ts[i + 2] - ts[i + 1]) + (cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]))*0.5f;
				}
				return Hermite(cps[i], vi, ts[i], cps[i + 1], vii, ts[i + 1], t);
				
			}
		}
	}
};

struct Material {
	vec3 ka, kd, ks;
	float shine;
	Material() {}
	Material(vec3 ka, vec3 kd, vec3 ks, float s) {
		this->ka = ka;
		this->kd = kd;
		this->ks = ks;
		this->shine = s;
	}
};

struct Texture {
	unsigned int textureId;
	Texture() {
		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		int width, height;
		width = 2;
		height = 2;
		float image[2 * 2 * 3] = { 1,0,0, 0,1,0, 0,1,0, 1,0,0 };
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
			0, GL_RGB, GL_FLOAT, image); //Texture -> OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

struct Light {
	vec3 La, Le;
	vec3 wLightPos;
	Light() {}
	Light(vec3 La, vec3 Le, vec3 wLightPos) {
		this->La = La;
		this->Le = Le;
		this->wLightPos = wLightPos;
	}
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
		checkLinking(shaderProgram);
	}
	virtual
		void Bind(RenderState& state) { glUseProgram(shaderProgram); }
};

class ShadowShader : public Shader {
	const char * vsSrc = R"(
        #version 330
        precision highp float;
		uniform mat4 MVP;
		layout(location = 0) in vec3 vtxPos;
		void main() { gl_Position = vec4(vtxPos, 1) * MVP; }
)";

	const char * fsSrc = R"(
        #version 330
        precision highp float;
		out vec4 fragmentColor;
		void main() { fragmentColor = vec4(1, 1, 1, 1); }
)";
public:
	ShadowShader() {
		Create(vsSrc, fsSrc, "fragmentColor");
	}

	void Bind(RenderState& state) {
		glUseProgram(shaderProgram);
		mat4 MVP = state.M * state.V * state.P;
		MVP.SetUniform(shaderProgram, "MVP");
	}
};
Material diff(vec3(1, 1, 1), vec3(0.2, 0.3, 1), vec3(1, 1, 1), 50.0f);
Material sandDiff(vec3(1, 1, 1), vec3(0.545, 0.271, 0.075), vec3(0.5, 0.5, 0.5), 20.0f);
Material redDiff(vec3(1, 1, 1), vec3(1, 0.3, 0.2), vec3(0.8, 0.8, 0.8), 30.0f);
Material snakeDiff1(vec3(1, 1, 1), vec3(0.3, 1, 0.2), vec3(1, 1, 1), 80.0f);
Material snakeDiff2(vec3(1, 1, 1), vec3(0.6, 1, 0.2), vec3(1, 1, 1), 80.0f);
class TexturedPhongShader : public Shader {
	const char * vsSrc = R"(
	#version 330
	precision highp float;
	uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
	uniform vec4  wLiPos;       // pos of light source 
	uniform vec3  wEye;         // pos of eye

	layout(location = 0) in vec3 vtxPos; // pos in model sp
	layout(location = 1) in vec3 vtxNorm;// normal in mod sp
	layout(location = 2) in vec2 vtxUV;

	out vec3 wNormal;           // normal in world space
	out vec3 wView;             // view in world space
	out vec3 wLight;            // light dir in world space
	out vec2 texcoord;

	void main() {
		gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		texcoord = vtxUV;

		vec4 wPos = vec4(vtxPos, 1) * M;
		wLight  = wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w;
		wView   = wEye * wPos.w - wPos.xyz;
		wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
	}
)";

	const char * fsSrc = R"(
	#version 330
	precision highp float;
	uniform vec3 kd, ks, ka;// diffuse, specular, ambient ref
	uniform vec3 La, Le;    // ambient and point source rad
	uniform float shine;    // shininess for specular ref
	uniform sampler2D samplerUnit;

	in  vec3 wNormal;       // interpolated world sp normal
	in  vec3 wView;         // interpolated world sp view
	in  vec3 wLight;        // interpolated world sp illum dir
	in vec2 texcoord;

	out vec4 fragmentColor; // output goes to frame buffer

	void main() {
		vec3 N = normalize(wNormal);
		vec3 V = normalize(wView);  
		vec3 L = normalize(wLight);
		vec3 H = normalize(L + V);
		float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
		vec3 color = ka * La + 
			       (kd*texture(samplerUnit,texcoord).xyz * cost + ks * pow(cosd,shine)) * Le;
		fragmentColor = vec4(color, 1);
	}
)";
public:
	TexturedPhongShader() {
		Create(vsSrc, fsSrc, "fragmentColor");
	}

	void Bind(RenderState& state) {
		glUseProgram(shaderProgram);
		mat4 MVP = state.M * state.V * state.P;
		mat4 M = state.M;
		mat4 Minv = state.Minv;
		vec4 wLiPos = vec4(state.light.wLightPos.x, state.light.wLightPos.y, state.light.wLightPos.z, 1.0f);
		vec3 wEye = state.wEye;
		MVP.SetUniform(shaderProgram, "MVP");
		M.SetUniform(shaderProgram, "M");
		Minv.SetUniform(shaderProgram, "Minv");
		wLiPos.SetUniform(shaderProgram, "wLiPos");
		wEye.SetUniform(shaderProgram, "wEye");
		vec3 kd = state.material->kd;
		vec3 ks = state.material->ks;
		vec3 ka = state.material->ka;
		vec3 La = state.light.La;
		vec3 Le = state.light.Le;
		float shine = state.material->shine;
		kd.SetUniform(shaderProgram, "kd");
		ks.SetUniform(shaderProgram, "ks");
		ka.SetUniform(shaderProgram, "ka");
		La.SetUniform(shaderProgram, "La");
		Le.SetUniform(shaderProgram, "Le");
		int loc = glGetUniformLocation(shaderProgram, "shine");
		glUniform1f(loc, shine);
	}
};

class PhongShader : public Shader {
	const char * vsSrc = R"(
	#version 330
	precision highp float;
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
	#version 330
	precision highp float;
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
		Create(vsSrc, fsSrc, "fragmentColor");
	}

	void Bind(RenderState& state) {
		glUseProgram(shaderProgram);
		mat4 MVP = state.M * state.V * state.P;
		mat4 M = state.M;
		mat4 Minv = state.Minv;
		vec4 wLiPos = vec4(state.light.wLightPos.x, state.light.wLightPos.y, state.light.wLightPos.z, 1.0f);
		vec3 wEye = state.wEye;
		MVP.SetUniform(shaderProgram, "MVP");
		M.SetUniform(shaderProgram, "M");
		Minv.SetUniform(shaderProgram, "Minv");
		wLiPos.SetUniform(shaderProgram, "wLiPos");
		wEye.SetUniform(shaderProgram, "wEye");
		vec3 kd = state.material->kd;
		vec3 ks = state.material->ks;
		vec3 ka = state.material->ka;
		vec3 La = state.light.La;
		vec3 Le = state.light.Le;
		float shine = state.material->shine;
		kd.SetUniform(shaderProgram, "kd");
		ks.SetUniform(shaderProgram, "ks");
		ka.SetUniform(shaderProgram, "ka");
		La.SetUniform(shaderProgram, "La");
		Le.SetUniform(shaderProgram, "Le");
		int loc = glGetUniformLocation(shaderProgram, "shine");
		glUniform1f(loc, shine);
	}
};

struct Geometry {
	unsigned int vao, nVtx;

	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	void Draw() {
		glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, nVtx);
	}
};

struct VertexData {
	vec3 position, normal;
	float u, v;
};

struct ParamSurface : Geometry {
	unsigned int vbo;
	virtual VertexData GenVertexData(float u, float v) = 0;
	void Create(int N, int M);
	void GenerateVbo(int N, int M) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

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
			nVtx * sizeof(VertexData), vtxData, GL_DYNAMIC_DRAW);
	}
};

void ParamSurface::Create(int N, int M) {
	nVtx = N * M * 6;
	glGenBuffers(1, &vbo);
	
	GenerateVbo(N, M);

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
		Create(256, 256); // tessellation level
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
};

class Snake : public ParamSurface {
	float radius;
	CatmullRom* cr;
public:
	Snake(CatmullRom* cr) : cr(cr) {
		radius = 1.0f;
		Create(32, 32); // tessellation level
	}

	VertexData GenVertexData(float u, float v) {

		VertexData vd;
		vec4 r(0, 0, -1, 1);
		r = r*Rotate(u*2*M_PI, 0, 1, 0);
		vd.normal = vec3(r.v[0], r.v[1], r.v[2]);
		vd.position = vd.normal*radius + cr->r(v);
		vd.u = u; vd.v = v;
		return vd;
	}
	vec3 GetLastCp() {
		return cr->cps[cr->cps.size() - 1];
	}
	void setCatmullrom(CatmullRom* cr) {
		delete this->cr;
		this->cr = cr;
	}
};

class Plank : public ParamSurface {
	vec3 n, r0;
	float a, b;
public:
	Plank(vec3 n, vec3 r, float a, float b) : n(n), r0(r), a(a), b(b) {
		Create(32, 32); // tessellation level
	}
	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = n;
		vd.position = vec3(r0.x + u*a, r0.y , r0.z - v*b);
		vd.u = u; vd.v = v;
		return vd;
	}
};

class Sand : public ParamSurface {
	vec3 r0;
public:
	Sand(vec3 r): r0(r){
		Create(256, 256); // tessellation level
	}
	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vec3 res;
		float x, z;
		x = 300;
		z = 300;
		res = vec3(u*x+r0.x,r0.y +(10 * abs(sinf(16 * u*M_PI))) + (10 * abs(cosf(16 * v *M_PI))),r0.z - v*z);
		vd.normal = vec3(0, 3 + sinf(8 * u*M_PI) + 5 + cosf(16 * v *M_PI),0);
		vd.position = res;
		vd.u = u; vd.v = v;
		return vd;
	}
};

struct Camera {
	vec3  wEye, wLookat, wVup;
	float fov, asp, fp, bp;
	Camera() {}
	Camera(vec3 wEye, vec3 wLookat, vec3 wVup, float fov, float asp, float fp, float bp) {
		this->wEye = wEye;
		this->wLookat = wLookat;
		this->wVup = wVup;
		this->fov = fov;
		this->asp = asp;
		this->fp = fp;
		this->bp = bp;
	}
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

struct Object {
	Shader *   shader;
	Material * material;
	Texture *  texture;
	Geometry * geometry;
	vec3 scale, pos, rotAxis;
	float rotAngle;
public:
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
	virtual void Animate(float t) {}
};

struct SpecificSphere : public Object {

};

struct SpecificSnake : public Object {
	virtual void Draw(RenderState state) {
		int samplerUnit = 0;
		int location = glGetUniformLocation(shader->shaderProgram, "samplerUnit");
		glUniform1i(location, samplerUnit);
		glActiveTexture(GL_TEXTURE0 + samplerUnit);
		glBindTexture(GL_TEXTURE_2D, texture->textureId);
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
	virtual void Animate(float dt) {
		CatmullRom* cr = new CatmullRom();
			
	}
};

struct SpecificPlank : public Object {

	virtual void Draw(RenderState state) {
		int samplerUnit = 0;
		int location = glGetUniformLocation(shader->shaderProgram, "samplerUnit");
		glUniform1i(location, samplerUnit);
		glActiveTexture(GL_TEXTURE0 + samplerUnit);
		glBindTexture(GL_TEXTURE_2D, texture->textureId);
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

};

struct SpecificSand : public Object {

};

struct Scene {
	Camera camera;
	std::vector<Object *> objects;
	Light light;
	RenderState state;
public:
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
};

void initCamera(Camera* camera) {
	camera->wEye = vec3(0, 0, 60);
	camera->wLookat = vec3(0, 0, 0);
	camera->wVup = vec3(0, 1, 0);
	camera->fov = 3.14 / 3;
	camera->asp = 1.0f;
	camera->fp = 2.0f;
	camera->bp = 100.0f;
}
void initLight(Light *l) {
	l->La = vec3(0.1, 0.1, 0.1);
	l->Le = vec3(1, 1, 1);
	l->wLightPos = vec3(2, 2, 2);
}
void initMaterial(Material* m) {
	m->ka = vec3(0, 1, 0);
	m->kd = vec3(1, 0, 0);
	m->ks = vec3(1, 1, 1);
	m->shine = 20;
}
// handle of the shader program
//unsigned int shaderProgram;
Scene scene;
Camera cam(vec3(30, 30, 30), vec3(0, 0, 0), vec3(0, 1, 0), 3.14f / 4.3, 1.0f, 2.0f, 100.0f);
Light light(vec3(0.1, 0.1, 0.1), vec3(1, 1, 1), vec3(40, 8, -5));
CatmullRom cr1(vec3(0,-1,0),vec3(0,-1,-1));
CatmullRom cr2(vec3(0, -1, 0), vec3(0, -1, -1));




// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glEnable(GL_DEPTH_TEST); // z-buffer is on
	glDisable(GL_CULL_FACE); // backface culling is off


	cr1.addControlPoint(vec3(7,25,-5), 0.0f);
	cr1.addControlPoint(vec3(10, 17, -10), 0.2f);
	cr1.addControlPoint(vec3(8, 12, -8), 0.4f);
	cr1.addControlPoint(vec3(7, 10, -10), 0.6f);
	cr1.addControlPoint(vec3(7, 7, -8), 0.8f);
	cr1.addControlPoint(vec3(8 , 4, -5), 1.0f);

	cr2.addControlPoint(vec3(7, 25, 15), 0.0f);
	cr2.addControlPoint(vec3(10, 17, 10), 0.2f);
	cr2.addControlPoint(vec3(8, 12, 12), 0.4f);
	cr2.addControlPoint(vec3(7, 10, 10), 0.6f);
	cr2.addControlPoint(vec3(7, 7, 12), 0.8f);
	cr2.addControlPoint(vec3(8, 4, 10), 1.0f);

	Texture* texture1 = new Texture();

	Sphere* sphere = new Sphere(vec3(0, 0, 0), 2.0f);
	

	Snake* snake1 = new Snake(&cr1);
	Snake* snake2 = new Snake(&cr2);

	vec3 h1 = snake1->GetLastCp();
	float r = 1.5f;
	//h1.y -= 2*r;
	Sphere* snakeHead1 = new Sphere(vec3(0,0,0), r);

	vec3 h2 = snake2->GetLastCp();
	//h2.y -= 2 * r;
	Sphere* snakeHead2 = new Sphere(vec3(0, 0, 0), r);

	Plank* plank = new Plank(vec3(0,1,0), vec3(0, 0, 40), 15.0f, 120.0f);
	Sand* sand = new Sand(vec3(-140, -20, 40));

	PhongShader* pshader = new PhongShader();
	TexturedPhongShader* tpshader = new TexturedPhongShader();

	SpecificSphere* specSphere = new SpecificSphere();
	SpecificSphere* specSnakeHead1 = new SpecificSphere();
	SpecificSphere* specSnakeHead2 = new SpecificSphere();
	SpecificSnake* specificSnake1 = new SpecificSnake();
	SpecificSnake* specificSnake2 = new SpecificSnake();
	SpecificPlank* specPlank = new SpecificPlank();
	SpecificSand* specificSand = new SpecificSand();

	specSphere->shader = pshader;
	specSphere->material = &redDiff;
	specSphere->geometry = sphere;
	specSphere->scale = vec3(1, 1, 1);
	specSphere->pos = vec3(8, 2, 22); // 8 0 22
	specSphere->rotAxis = vec3(0, 1, 0);
	specSphere->rotAngle = 0;

	specSnakeHead1->shader = pshader;
	specSnakeHead1->material = &diff;
	specSnakeHead1->geometry = snakeHead1;
	specSnakeHead1->scale = vec3(1, 2, 0.5);
	specSnakeHead1->pos = h1;
	specSnakeHead1->rotAxis = vec3(0, 1, 0);
	specSnakeHead1->rotAngle = 0;

	specSnakeHead2->shader = pshader;
	specSnakeHead2->material = &diff;
	specSnakeHead2->geometry = snakeHead2;
	specSnakeHead2->scale = vec3(1, 2, 0.5);
	specSnakeHead2->pos = h2;
	specSnakeHead2->rotAxis = vec3(0, 1, 0);
	specSnakeHead2->rotAngle = 0;

	specPlank->shader = tpshader;
	specPlank->material = &diff;
	specPlank->texture = texture1;
	specPlank->geometry = plank;
	specPlank->scale = vec3(1, 1, 1);
	specPlank->pos = vec3(0, 0, 0);
	specPlank->rotAxis = vec3(0, 0, 1);
	specPlank->rotAngle = 0;

	specificSand->shader = pshader;
	specificSand->material = &sandDiff;
	specificSand->geometry = sand;
	specificSand->scale = vec3(1, 1, 1);
	specificSand->pos = vec3(0, 0, 0);
	specificSand->rotAxis = vec3(0, 1, 0);
	specificSand->rotAngle = 0;

	specificSnake1->shader = tpshader;
	specificSnake1->material = &snakeDiff1;
	specificSnake1->texture = texture1;
	specificSnake1->geometry = snake1;
	specificSnake1->scale = vec3(1, 1, 1);
	specificSnake1->pos = vec3(0, 0, 0);
	specificSnake1->rotAxis = vec3(0, 1, 0);
	specificSnake1->rotAngle = 0;

	specificSnake2->shader = tpshader;
	specificSnake2->material = &snakeDiff2;
	specificSnake2->texture = texture1;
	specificSnake2->geometry = snake2;
	specificSnake2->scale = vec3(1, 1, 1);
	specificSnake2->pos = vec3(0, 0, 0);
	specificSnake2->rotAxis = vec3(0, 1, 0);
	specificSnake2->rotAngle = 0;

	scene.camera = cam;
	scene.light = light;
	scene.objects.push_back(specSphere);
	scene.objects.push_back(specSnakeHead1);
	scene.objects.push_back(specSnakeHead2);
	scene.objects.push_back(specificSand);
	scene.objects.push_back(specificSnake1);
	scene.objects.push_back(specificSnake2);
	scene.objects.push_back(specPlank);

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
