#version 330 core

in vec3 normal;

out vec4 FragColor;

void main()
{
    vec3 lightDir = normalize(vec3(2.0, 3.0, 1.0));
    float diff = max(dot(normalize(normal), lightDir), 0.1);
    FragColor = vec4(vec3(diff), 1.0);
}