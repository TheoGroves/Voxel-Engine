#version 330

in vec3 aPos;
in vec3 aNormal;
in vec2 uvCoords;
in float aAO;

out vec3 vNormal;
out vec3 vColor;
out vec2 uv;
out float ao;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main() {
    vNormal = aNormal;
    uv = uvCoords;
    ao = aAO;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}