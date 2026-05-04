#version 330 core

in vec3 vColor;
in vec3 vNormal;

out vec4 FragColor;

void main()
{
    vec3 lightDir = normalize(vec3(2.0, 3.0, 1.0));
    float diff = max(dot(normalize(vNormal), lightDir), 0.3);
    FragColor = vec4(vColor * diff, 1.0);
}