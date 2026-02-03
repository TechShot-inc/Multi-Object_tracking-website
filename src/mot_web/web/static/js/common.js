function showModal(message) {
    console.log('Showing modal:', message);
    document.getElementById('modal-message').textContent = message;
    document.getElementById('modal').classList.remove('hidden');
}

async function getImageData(imgElement) {
    return new Promise((resolve, reject) => {
        if (!imgElement.src || imgElement.src.includes('data:image')) {
            resolve(imgElement.src);
            return;
        }
        const canvas = document.createElement('canvas');
        canvas.width = imgElement.naturalWidth;
        canvas.height = imgElement.naturalHeight;
        const ctx = canvas.getContext('2d');
        imgElement.crossOrigin = 'Anonymous';
        imgElement.onload = () => {
            ctx.drawImage(imgElement, 0, 0);
            resolve(canvas.toDataURL('image/png'));
        };
        imgElement.onerror = () => reject(new Error('Failed to load image'));
        imgElement.src = imgElement.src;
    });
}