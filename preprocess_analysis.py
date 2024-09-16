
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem de câncer de mama
image = cv2.imread('breast_cancer_image.png', 0)  # 0 para carregar em grayscale

# Equalização de histograma
equalized_image = cv2.equalizeHist(image)

# Salvar e exibir a imagem pré-processada
cv2.imwrite('preprocessed_image.png', equalized_image)

plt.imshow(equalized_image, cmap='gray')
plt.title('Imagem Pré-processada')
plt.show()

# Definir um limite para detectar pixels "claros" (acima de um certo valor)
threshold_value = 200  # valor pode ser ajustado dependendo da imagem
_, binary_image = cv2.threshold(equalized_image, threshold_value, 255, cv2.THRESH_BINARY)

# Contar pixels claros (onde há tumor ou alta densidade)
white_pixels_count = np.sum(binary_image == 255)
total_pixels = image.size
percentage_of_white = (white_pixels_count / total_pixels) * 100

# Inferir a presença de câncer baseado no número de pixels brancos
if percentage_of_white > 5:  # Limite arbitrário de 5% de pixels claros
    result = "Câncer detectado"
else:
    result = "Ausente"

# Exibir resultado
print(f"Porcentagem de pixels claros: {percentage_of_white:.2f}%")
print(f"Resultado da análise: {result}")

# Adicionar resultado na imagem
cv2.putText(binary_image, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imwrite('classification_output.png', binary_image)

# Exibir a imagem com a classificação
cv2.imshow('Classificação', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
