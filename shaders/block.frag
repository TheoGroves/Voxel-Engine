#version 330 core

in vec3 vNormal;
in vec2 uv;

out vec4 FragColor;

uniform sampler2D atlas;

void main()
{
    vec3 lightDir = normalize(vec3(2.0, 3.0, 1.0));
    float diff = max(dot(normalize(vNormal), lightDir), 0.3);
    FragColor = vec4(texture(atlas, uv) * diff);
}