#version 330 core

in vec3 normal;

out vec4 FragColor;

void main()
{
    vec3 lightDir = normalize(vec3(2.0, 3.0, 1.0));
    float diff = max(dot(normalize(normal), lightDir), 0.3);
    FragColor = vec4(0.412*diff, 0.353*diff, 0.29*diff, 1.0);
}