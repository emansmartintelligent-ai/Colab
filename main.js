// Botanical Garden Festival - Main JavaScript

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeAnimations();
    initializeCarousels();
    initializeParticles();
    initializeQuiz();
});

// Navigation function
function navigateTo(page) {
    // Add loading animation
    anime({
        targets: 'body',
        opacity: [1, 0.8],
        duration: 300,
        easing: 'easeOutQuad',
        complete: function() {
            window.location.href = page;
        }
    });
}

// Initialize entrance animations
function initializeAnimations() {
    // Hero title animation
    anime({
        targets: '#heroTitle',
        opacity: [0, 1],
        translateY: [50, 0],
        duration: 1000,
        delay: 500,
        easing: 'easeOutExpo'
    });

    // Hero subtitle animation
    anime({
        targets: '#heroSubtitle',
        opacity: [0, 1],
        translateY: [30, 0],
        duration: 800,
        delay: 800,
        easing: 'easeOutExpo'
    });

    // Hero buttons animation
    anime({
        targets: '#heroButtons',
        opacity: [0, 1],
        translateY: [20, 0],
        duration: 600,
        delay: 1100,
        easing: 'easeOutExpo'
    });

    // Animate cards on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                anime({
                    targets: entry.target,
                    opacity: [0, 1],
                    translateY: [30, 0],
                    duration: 600,
                    easing: 'easeOutExpo'
                });
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all cards
    document.querySelectorAll('.card-hover').forEach(card => {
        card.style.opacity = '0';
        observer.observe(card);
    });
}

// Initialize carousels
function initializeCarousels() {
    if (document.getElementById('schedulePreview')) {
        new Splide('#schedulePreview', {
            type: 'loop',
            perPage: 3,
            perMove: 1,
            gap: '1rem',
            autoplay: true,
            interval: 4000,
            pauseOnHover: true,
            breakpoints: {
                768: {
                    perPage: 1,
                },
                1024: {
                    perPage: 2,
                }
            }
        }).mount();
    }
}

// Initialize particle system
function initializeParticles() {
    if (document.getElementById('particles')) {
        new p5((p) => {
            let particles = [];
            const numParticles = 30;

            p.setup = function() {
                const canvas = p.createCanvas(p.windowWidth, p.windowHeight);
                canvas.parent('particles');
                canvas.style('position', 'absolute');
                canvas.style('top', '0');
                canvas.style('left', '0');
                canvas.style('z-index', '1');
                canvas.style('pointer-events', 'none');

                // Create particles
                for (let i = 0; i < numParticles; i++) {
                    particles.push({
                        x: p.random(p.width),
                        y: p.random(p.height),
                        size: p.random(2, 6),
                        speedX: p.random(-0.5, 0.5),
                        speedY: p.random(-0.8, -0.2),
                        opacity: p.random(0.1, 0.3)
                    });
                }
            };

            p.draw = function() {
                p.clear();
                
                // Update and draw particles
                particles.forEach(particle => {
                    // Update position
                    particle.x += particle.speedX;
                    particle.y += particle.speedY;

                    // Reset particle if it goes off screen
                    if (particle.y < -10) {
                        particle.y = p.height + 10;
                        particle.x = p.random(p.width);
                    }
                    if (particle.x < -10 || particle.x > p.width + 10) {
                        particle.x = p.random(p.width);
                    }

                    // Draw particle
                    p.fill(255, 255, 255, particle.opacity * 255);
                    p.noStroke();
                    p.ellipse(particle.x, particle.y, particle.size);
                });
            };

            p.windowResized = function() {
                p.resizeCanvas(p.windowWidth, p.windowHeight);
            };
        });
    }
}

// Quiz functionality
let currentQuiz = {
    questions: [
        {
            question: "Which plant family includes both tomatoes and potatoes?",
            options: ["Solanaceae", "Rosaceae", "Fabaceae", "Asteraceae"],
            correct: 0,
            fact: "The Solanaceae family, also known as the nightshade family, includes many important food crops like tomatoes, potatoes, peppers, and eggplants."
        },
        {
            question: "What is the process by which plants convert sunlight into energy?",
            options: ["Respiration", "Photosynthesis", "Transpiration", "Germination"],
            correct: 1,
            fact: "Photosynthesis is the remarkable process where plants use chlorophyll to convert sunlight, water, and carbon dioxide into glucose and oxygen."
        },
        {
            question: "Which part of the plant is responsible for water absorption?",
            options: ["Leaves", "Stems", "Roots", "Flowers"],
            correct: 2,
            fact: "Plant roots have specialized structures called root hairs that dramatically increase surface area for efficient water and nutrient absorption."
        },
        {
            question: "What is the world's largest flower?",
            options: ["Rafflesia arnoldii", "Titan arum", "Sunflower", "Lotus"],
            correct: 0,
            fact: "Rafflesia arnoldii, found in Southeast Asia, can grow up to 3 feet across and weigh up to 15 pounds, making it the largest flower in the world."
        },
        {
            question: "Which gas do plants release during photosynthesis?",
            options: ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"],
            correct: 2,
            fact: "During photosynthesis, plants release oxygen as a byproduct, which is essential for animal life on Earth."
        }
    ],
    currentQuestion: 0,
    score: 0,
    started: false
};

function initializeQuiz() {
    // Quiz is initialized but not started until user clicks
}

function startQuiz() {
    currentQuiz.started = true;
    currentQuiz.currentQuestion = 0;
    currentQuiz.score = 0;
    showQuestion();
}

function showQuestion() {
    const quiz = currentQuiz;
    const question = quiz.questions[quiz.currentQuestion];
    const quizContent = document.getElementById('quizContent');
    
    quizContent.innerHTML = `
        <div class="text-center mb-6">
            <div class="flex justify-center items-center mb-4">
                <div class="w-12 h-12 bg-accent-green rounded-full flex items-center justify-center mr-4">
                    <span class="text-white font-semibold">${quiz.currentQuestion + 1}</span>
                </div>
                <div class="text-sm text-gray-500">Question ${quiz.currentQuestion + 1} of ${quiz.questions.length}</div>
            </div>
            <h3 class="text-xl font-semibold mb-6" style="color: var(--primary-green);">${question.question}</h3>
        </div>
        
        <div class="space-y-3 mb-6">
            ${question.options.map((option, index) => `
                <button class="quiz-option w-full p-4 text-left rounded-xl border-2 border-gray-200 hover:border-secondary-green transition-all duration-300" 
                        onclick="selectAnswer(${index})" style="border-color: #e5e7eb;">
                    ${option}
                </button>
            `).join('')}
        </div>
    `;
}

function selectAnswer(selectedIndex) {
    const quiz = currentQuiz;
    const question = quiz.questions[quiz.currentQuestion];
    const isCorrect = selectedIndex === question.correct;
    
    if (isCorrect) {
        quiz.score++;
    }
    
    // Show feedback
    showFeedback(selectedIndex, question.correct, question.fact, isCorrect);
}

function showFeedback(selectedIndex, correctIndex, fact, isCorrect) {
    const quizContent = document.getElementById('quizContent');
    const options = document.querySelectorAll('.quiz-option');
    
    // Highlight correct and incorrect answers
    options.forEach((option, index) => {
        if (index === correctIndex) {
            option.style.borderColor = 'var(--secondary-green)';
            option.style.backgroundColor = 'rgba(74, 124, 89, 0.1)';
        } else if (index === selectedIndex && !isCorrect) {
            option.style.borderColor = '#ef4444';
            option.style.backgroundColor = 'rgba(239, 68, 68, 0.1)';
        }
        option.disabled = true;
    });
    
    setTimeout(() => {
        quizContent.innerHTML = `
            <div class="text-center">
                <div class="w-16 h-16 ${isCorrect ? 'bg-secondary-green' : 'bg-red-500'} rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        ${isCorrect ? 
                            '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>' :
                            '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>'
                        }
                    </svg>
                </div>
                <h3 class="text-xl font-semibold mb-4" style="color: var(--primary-green);">
                    ${isCorrect ? 'Correct!' : 'Not quite right'}
                </h3>
                <p class="text-gray-600 mb-6">${fact}</p>
                <button class="btn-primary px-6 py-3 rounded-full font-semibold" onclick="nextQuestion()">
                    ${currentQuiz.currentQuestion < currentQuiz.questions.length - 1 ? 'Next Question' : 'See Results'}
                </button>
            </div>
        `;
    }, 1000);
}

function nextQuestion() {
    const quiz = currentQuiz;
    quiz.currentQuestion++;
    
    if (quiz.currentQuestion < quiz.questions.length) {
        showQuestion();
    } else {
        showResults();
    }
}

function showResults() {
    const quiz = currentQuiz;
    const percentage = Math.round((quiz.score / quiz.questions.length) * 100);
    const quizContent = document.getElementById('quizContent');
    
    let message = '';
    let emoji = '';
    
    if (percentage >= 80) {
        message = 'Excellent! You have a green thumb and deep botanical knowledge.';
        emoji = '🌿';
    } else if (percentage >= 60) {
        message = 'Great job! You know your plants well and have a solid foundation.';
        emoji = '🌱';
    } else if (percentage >= 40) {
        message = 'Good effort! There is room to grow your plant knowledge.';
        emoji = '🌾';
    } else {
        message = 'Keep learning! Our festival workshops will help you bloom.';
        emoji = '🌸';
    }
    
    quizContent.innerHTML = `
        <div class="text-center">
            <div class="text-6xl mb-4">${emoji}</div>
            <h3 class="text-2xl font-semibold mb-4" style="color: var(--primary-green);">
                Quiz Complete!
            </h3>
            <div class="text-4xl font-bold mb-4" style="color: var(--secondary-green);">
                ${quiz.score}/${quiz.questions.length}
            </div>
            <div class="text-xl mb-6" style="color: var(--earth-tone);">
                ${percentage}%
            </div>
            <p class="text-gray-600 mb-6">${message}</p>
            <div class="flex flex-col sm:flex-row gap-4 justify-center">
                <button class="btn-primary px-6 py-3 rounded-full font-semibold" onclick="startQuiz()">
                    Try Again
                </button>
                <button class="btn-secondary px-6 py-3 rounded-full font-semibold" onclick="navigateTo('workshops.html')">
                    Join Workshops
                </button>
            </div>
        </div>
    `;
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg text-white ${
        type === 'success' ? 'bg-secondary-green' : 
        type === 'error' ? 'bg-red-500' : 
        'bg-blue-500'
    }`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    anime({
        targets: notification,
        translateX: [300, 0],
        opacity: [0, 1],
        duration: 300,
        easing: 'easeOutExpo'
    });
    
    setTimeout(() => {
        anime({
            targets: notification,
            translateX: [0, 300],
            opacity: [1, 0],
            duration: 300,
            easing: 'easeInExpo',
            complete: () => {
                document.body.removeChild(notification);
            }
        });
    }, 3000);
}

// Handle form submissions and interactions
function handleWorkshopBooking(workshopName) {
    showNotification(`Booking confirmed for ${workshopName}!`, 'success');
}

function handleScheduleFilter(filterType) {
    showNotification(`Filtering by: ${filterType}`, 'info');
}

// Smooth scrolling for internal links
document.addEventListener('click', function(e) {
    if (e.target.matches('a[href^="#"]')) {
        e.preventDefault();
        const target = document.querySelector(e.target.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
});

// Handle window resize for responsive particles
window.addEventListener('resize', function() {
    // Particle system will handle its own resize
});

// Add loading states for better UX
function addLoadingState(element) {
    element.style.opacity = '0.6';
    element.style.pointerEvents = 'none';
}

function removeLoadingState(element) {
    element.style.opacity = '1';
    element.style.pointerEvents = 'auto';
}

// Initialize tooltips and helpful hints
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

function showTooltip(e) {
    const tooltip = document.createElement('div');
    tooltip.className = 'absolute bg-gray-800 text-white text-sm px-2 py-1 rounded shadow-lg z-50';
    tooltip.textContent = e.target.getAttribute('data-tooltip');
    tooltip.style.bottom = '100%';
    tooltip.style.left = '50%';
    tooltip.style.transform = 'translateX(-50%)';
    tooltip.style.marginBottom = '5px';
    
    e.target.style.position = 'relative';
    e.target.appendChild(tooltip);
}

function hideTooltip(e) {
    const tooltip = e.target.querySelector('.absolute.bg-gray-800');
    if (tooltip) {
        e.target.removeChild(tooltip);
    }
}

// Performance optimization
function optimizeImages() {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.loading = 'lazy';
        img.addEventListener('load', function() {
            this.style.opacity = '1';
        });
    });
}

// Initialize performance optimizations
document.addEventListener('DOMContentLoaded', function() {
    optimizeImages();
    initializeTooltips();
});

// Export functions for global use
window.navigateTo = navigateTo;
window.startQuiz = startQuiz;
window.selectAnswer = selectAnswer;
window.nextQuestion = nextQuestion;
window.handleWorkshopBooking = handleWorkshopBooking;
window.handleScheduleFilter = handleScheduleFilter;