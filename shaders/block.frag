#version 330 core

in vec3 vNormal;
in vec2 uv;
in float ao;

out vec4 FragColor;

uniform sampler2D atlas;

void main()
{
    vec3 lightDir = normalize(vec3(2.0, 3.0, 1.0));
    float diff = min(max(dot(normalize(vNormal), lightDir)*2, 0.6), 1.0);
    float ao_light = 0.4 + 0.6 * ao;
    FragColor = vec4(texture(atlas, uv) * diff * ao);
}