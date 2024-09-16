
import cv2
import matplotlib.pyplot as plt

# Carregar imagem
image = cv2.imread('input_image.png')

# Separar canais RGB
blue, green, red = cv2.split(image)

# Salvar e exibir os canais
cv2.imwrite('red_channel.png', red)
cv2.imwrite('green_channel.png', green)
cv2.imwrite('blue_channel.png', blue)

# Exibir os resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(red, cmap='Reds')
plt.title('Canal Vermelho')

plt.subplot(1, 3, 2)
plt.imshow(green, cmap='Greens')
plt.title('Canal Verde')

plt.subplot(1, 3, 3)
plt.imshow(blue, cmap='Blues')
plt.title('Canal Azul')

plt.show()
