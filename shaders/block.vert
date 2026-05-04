#version 330

in vec3 aPos;
in vec3 aNormal;
in vec3 aColor;

out vec3 vNormal;
out vec3 vColor;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main() {
    vNormal = aNormal;
    vColor = aColor;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}